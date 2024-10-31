import argparse
import logging
import os
from os.path import join, exists
from tqdm import tqdm
import numpy as np
import pyvista as pv
import pandas as pd
import torch

from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from torchmetrics.wrappers import MetricTracker

from shape_utils import ShapeDataset, metric_collection, calc_loss
from shapecentral.networks import delta_conv


def run_classification_model(data_dir, info_df, res_dir, label_name='labels',
                             n_fold=5, split_name_base='split_', label_smoothing=.2,
                             n_class=6, input_features='k', width=128, depth=4, n_points=1024,
                             num_neighbors=20, grad_regularizer=0.001, grad_kernel_width=1,
                             n_epoch=200, lr=1e-3, scheduler_step=50, scheduler_gamma=.5,
                             device_name='cuda'):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%d/%m/%Y %H:%M:%S')

    # Dealing with device
    if device_name == 'mps':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    elif device_name == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    logging.info(f'Using device {device}')

    possible_features = ['xyz', 'hks', 'k', 'k1k2', 'k1k2n', 'shot', 'shot16']

    if input_features in ['xyz']:
        c_in = 3
    elif input_features in ['k']:
        c_in = 1
    elif input_features in ['k1k2', 'k1k2n']:
        c_in = 2
    elif input_features in ['shot16', 'hks']:
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

    n_fold_train_ids = [np.argwhere(info_df[splits[i]] == 1).flatten() for i in range(n_fold)]
    n_fold_test_ids = [np.argwhere(info_df[splits[i]] == 0).flatten() for i in range(n_fold)]

    labels = info_df[label_name].to_list()
    all_shapes = [pv.read(join(data_dir, str(i) + '.vtk')) for i in info_df.index]

    pre_transform = Compose([delta_conv.transforms.NormalizeArea(),
                             delta_conv.transforms.NormalizeAxes(), ])

    full_dataset = ShapeDataset(data_list=all_shapes, labels=labels, input_name=input_features,
                                pre_transform=pre_transform, n_points=n_points)

    if isinstance(width, list):
        channels = width
    else:
        channels = [width] * depth

    for i in range(n_fold):
        res_folder = res_folders[i]

        train_subset = Subset(full_dataset, n_fold_train_ids[i])
        train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
        test_subset = Subset(full_dataset, n_fold_test_ids[i])
        test_loader = DataLoader(test_subset)

        # model and optimizer
        model = delta_conv.models.DeltaNetClassification(in_channels=c_in,
                                                         num_classes=n_class,
                                                         conv_channels=channels,
                                                         num_neighbors=num_neighbors,
                                                         grad_regularizer=grad_regularizer,
                                                         grad_kernel_width=grad_kernel_width)
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

        logging.info('Starting training')

        metrics = metric_collection(n_classes=n_class, device=device)
        metrics_test = metrics.clone()
        tracker = MetricTracker(metrics)
        tracker_test = MetricTracker(metrics_test)
        loss_evol = []
        base_acc = .8
        for epoch in tqdm(range(n_epoch)):
            tracker.increment()
            tracker_test.increment()
            model.train()
            for data in train_loader:
                optimizer.zero_grad()
                data = data.to(device)
                preds = model(data)
                labels = data.y.long()
                loss = calc_loss(preds, labels, smoothing=True)
                loss.backward()
                optimizer.step()

                pred_labels = torch.argmax(preds, dim=-1)
                tracker.update(pred_labels, labels)
            scheduler.step()

            model.eval()
            with torch.no_grad():
                for data in test_loader:
                    data = data.to(device)
                    preds = model(data)
                    labels = data.y.long()
                    pred_labels = torch.argmax(preds, dim=-1)
                    tracker_test.update(pred_labels, labels)

                test_acc = tracker_test.compute()['acc'].item()
                if test_acc > base_acc:
                    torch.save(model.state_dict(), join(res_folder, 'best_model.pt'))
                base_acc = test_acc

            logging.info(f"Epoch {epoch}")
            logging.info(f"Train state: {tracker.compute()}")
            logging.info(f"Test state: {tracker_test.compute()}")
            logging.info(f"loss: {loss.item()}")
            loss_evol.append(loss.item())
        metric_results_all = tracker.compute_all()
        metric_test_results_all = tracker_test.compute_all()

        torch.save(loss_evol, join(res_folder, 'loss_evol.pt'))
        torch.save(metric_results_all, join(res_folder, 'all_train_metrics.pt'))
        torch.save(metric_test_results_all, join(res_folder, 'all_test_metrics.pt'))



def main(base_fold, dataset):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%d/%m/%Y %H:%M:%S')
    device = 'cuda'
    logging.info(f'Starting {dataset} experiment')
    data_fold = join(base_fold, dataset)
    data = join(data_fold, 'shapes')
    info = pd.read_csv(join(data_fold, 'info.csv'), index_col='idx', delimiter=';')
    res_fold = join(data_fold, 'results', 'delta_conv')
    if not exists(res_fold):
        os.mkdir(res_fold)
    split_name = '10_split_'
    if dataset == 'Shrec_16':
        n = 30
        n_pts = 2048
    else:
        n = 30
        n_pts = None

    for f in ['xyz', 'k1k2', 'hks', 'k', 'shot', 'shot16']:
        res = join(res_fold, f)
        run_classification_model(data_dir=data, info_df=info, res_dir=res, label_name='labels',
                                 n_fold=5, split_name_base=split_name, n_points=n_pts,
                                 n_class=n, input_features=f, width=32, depth=4,
                                 num_neighbors=20, grad_regularizer=0.001, grad_kernel_width=1,
                                 n_epoch=100, lr=1e-3, scheduler_step=50, scheduler_gamma=.5,
                                 device_name=device)
        logging.info(f'{f} done')
    logging.info(f'{dataset} experiment is done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    dataset = args.dataset

    base_fold = 'Datasets/Shapes'
    main(base_fold, dataset)
