import os
from os.path import join, exists
import logging
import pyvista as pv
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchmetrics.wrappers import MetricTracker
from tqdm import tqdm
from shapecentral.networks import diffusion_net
from shape_utils import shape_dataset, metric_collection


def run_classification_model(data_dir, info_df, res_dir, label_name='labels', cached_data=None,
                             n_fold=5, split_name_base='split_', random_rotate=False, label_smoothing=None,
                             n_class=6, input_features='k', width=128, depth=4, k_eig=128,
                             n_epoch=200, lr=1e-3, scheduler_step=50, scheduler_gamma=.5,
                             device_name='cuda'):
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%d/%m/%Y %H:%M:%S')

    # Dealing with device
    if device_name == 'mps':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        logging.debug("Metal doesn't support Sparse objects yet, will have to change them to dense")
    elif device_name == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    logging.debug(f'Using device {device}')

    possible_features = ['k', 'k1k2', 'xyz', 'shot16', 'hks', 'shot']
    if input_features == 'k':
        c_in = 1
    elif input_features in ['k1k2', 'k1k2n']:
        c_in = 2
    elif input_features == 'xyz':
        c_in = 3
    elif input_features in ['hks', 'shot16']:
        c_in = 16
    elif input_features == 'shot':
        c_in = 64
    else:
        raise ValueError(f'input_features must be in {possible_features}')

    # Dealing with result folders
    splits = [split_name_base + str(i) for i in range(n_fold)]
    if not exists(res_dir):
        os.mkdir(res_dir)
    if n_fold == 1:
        res_folders = [res_dir]
    else:
        res_folders = [join(res_dir, i) for i in splits]
        for f in res_folders:
            if not exists(f):
                os.mkdir(f)

    # Dealing with cache folders
    if cached_data is not None:
        if not exists(cached_data):
            save_to_cache = True
            use_cache = False
        else:
            save_to_cache = False
            use_cache = True
    else:
        save_to_cache = False
        use_cache = False

    n_fold_train_ids = [np.argwhere(info_df[splits[i]] == 1).flatten() for i in range(n_fold)]
    n_fold_test_ids = [np.argwhere(info_df[splits[i]] == 0).flatten() for i in range(n_fold)]

    labels = info_df[label_name].to_list()
    all_shapes = [pv.read(join(data_dir, str(i) + '.vtk')) for i in info_df.index]

    full_dataset = shape_dataset(data=all_shapes, labels=labels,
                                 representation_names=['k', 'k1k2', 'xyz', 'hks', 'shot'],
                                 k_eig=k_eig, normalise_rep=False, use_cache=use_cache, save_to_cache=save_to_cache,
                                 cache_path=cached_data)

    for i in range(n_fold):
        res_folder = res_folders[i]

        train_subset = Subset(full_dataset, n_fold_train_ids[i])
        train_loader = DataLoader(train_subset, batch_size=None, shuffle=True)
        test_subset = Subset(full_dataset, n_fold_test_ids[i])
        test_loader = DataLoader(test_subset, batch_size=None)

        # model and optimizer
        model = diffusion_net.layers.DiffusionNet(C_in=c_in,
                                                  C_out=n_class,
                                                  C_width=width,
                                                  N_block=depth,
                                                  last_activation=lambda x: torch.nn.functional.log_softmax(x, dim=-1),
                                                  outputs_at='global_mean',
                                                  dropout=False)

        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

        logging.debug('Starting training')

        metrics = metric_collection(n_classes=n_class, device=device).to(device)
        metrics_test = metrics.clone().to(device)
        tracker = MetricTracker(metrics).to(device)
        tracker_test = MetricTracker(metrics_test).to(device)
        loss_evol = []
        base_acc = .8

        for epoch in tqdm(range(n_epoch)):
            tracker.increment()
            tracker_test.increment()
            model.train()
            for data in train_loader:
                optimizer.zero_grad()
                features, verts, faces, frames, mass, L, evals, evecs, gradX, gradY, label = data
                k_reps, k1k2_reps, hks_reps, shot_reps = features

                # Move to device
                verts = verts.to(device)
                faces = faces.to(device)
                mass = mass.to(device)
                L = L.to(device)
                evals = evals.to(device)
                evecs = evecs.to(device)
                gradX = gradX.to(device)
                gradY = gradY.to(device)
                label = label.to(device)

                if input_features == 'xyz':
                    feats = verts
                    if random_rotate:
                        feats = diffusion_net.utils.random_rotate_points(feats)
                else:
                    if input_features == 'k':
                        features = k_reps
                    elif input_features == 'k1k2':
                        features = k1k2_reps
                    elif input_features == 'k1k2n':
                        features = k1k2_reps
                        features = features - torch.mean(features)
                    elif input_features == 'hks':
                        features = hks_reps
                    elif input_features == 'shot':
                        features = shot_reps
                        # features = features - torch.mean(features, dim=-2, keepdim=True)
                    elif input_features == 'shot16':
                        features = shot_reps
                        features = features.reshape((-1, 4, 4, 2, 2))[:, :2, :2, :, :].reshape((-1, 16))
                        features = torch.tensor(np.ascontiguousarray(features)).float()

                    if len(features.shape) == 1:
                        features = features[:, None]
                    feats = features
                feats = feats.to(device)

                preds = model(feats, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
                if label_smoothing is not None:
                    loss = diffusion_net.utils.label_smoothing_log_loss(preds, label, label_smoothing)
                else:
                    loss = torch.nn.functional.nll_loss(preds, label)
                loss.backward()
                optimizer.step()

                pred_label = torch.argmax(preds, dim=-1)

                tracker.update(torch.FloatTensor([pred_label]).to(device), torch.FloatTensor([label]).to(device))
            scheduler.step()

            model.eval()
            with torch.no_grad():
                for data in test_loader:
                    features, verts, faces, frames, mass, L, evals, evecs, gradX, gradY, label = data
                    k_reps, k1k2_reps, hks_reps, shot_reps = features
                    # Move to device
                    verts = verts.to(device)
                    faces = faces.to(device)
                    mass = mass.to(device)
                    if device_name == 'mps':
                        L = L.to_dense()
                        gradX = gradX.to_dense()
                        gradY = gradY.to_dense()
                    L = L.to(device)
                    evals = evals.to(device)
                    evecs = evecs.to(device)
                    gradX = gradX.to(device)
                    gradY = gradY.to(device)
                    label = label.to(device)

                    if input_features == 'xyz':
                        feats = verts
                        if random_rotate:
                            feats = diffusion_net.utils.random_rotate_points(feats)
                    else:
                        if input_features == 'k':
                            features = k_reps
                        elif input_features == 'k1k2':
                            features = k1k2_reps
                            # features = features - torch.mean(features)
                        elif input_features == 'k1k2n':
                            features = k1k2_reps
                            features = features - torch.mean(features)
                        elif input_features == 'hks':
                            features = hks_reps
                        elif input_features == 'shot':
                            features = shot_reps
                            # features = features - torch.mean(features, dim=-2, keepdim=True)
                        elif input_features == 'shot16':
                            features = shot_reps
                            features = features.reshape((-1, 4, 4, 2, 2))[:, :2, :2, :, :].reshape((-1, 16))
                            features = torch.tensor(np.ascontiguousarray(features)).float()

                        if len(features.shape) == 1:
                            features = features[:, None]
                        feats = features
                    feats = feats.to(device)

                    preds = model(feats, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
                    pred_label = torch.argmax(preds, dim=-1)
                    tracker_test.update(torch.FloatTensor([pred_label]).to(device),
                                        torch.FloatTensor([label]).to(device))

                test_acc = tracker_test.compute()['acc'].item()
                if test_acc > base_acc:
                    torch.save(model.state_dict(), join(res_folder, 'best_model.pt'))
                base_acc = test_acc

            logging.debug(f"Epoch {epoch}")
            logging.debug(f"Train state: {tracker.compute()}")
            logging.debug(f"Test state: {tracker_test.compute()}")

            loss_evol.append(loss.item())
        metric_results_all = tracker.compute_all()
        metric_test_results_all = tracker_test.compute_all()

        torch.save(loss_evol, join(res_folder, 'loss_evol.pt'))
        torch.save(metric_results_all, join(res_folder, 'all_train_metrics.pt'))
        torch.save(metric_test_results_all, join(res_folder, 'all_test_metrics.pt'))
