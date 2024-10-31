from os.path import join, exists
import pyvista as pv
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from shapecentral.networks import diffusion_net
from shape_utils import shape_dataset
import polyscope as ps


def run_segmentation_model(data_dir, info_df, weight_path, label_name='labels', random_rotate=True,
                           n_class=6, input_features='k', width=128, depth=4, k_eig=128, cached_data=None):

    device = torch.device('cpu')

    possible_features = ['k', 'k1k2', 'xyz', 'hks', 'shot']
    if input_features == 'k':
        c_in = 1
    elif input_features == 'k1k2':
        c_in = 2
    elif input_features in ['xyz']:
        c_in = 3
    elif input_features in ['hks', 'shot16']:
        c_in = 16
    elif input_features == 'shot':
        c_in = 64
    else:
        raise ValueError(f'input_features must be in {possible_features}')

    all_shapes = [pv.read(join(data_dir, str(i) + '.vtk')) for i in info_df.index]

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

    full_dataset = shape_dataset(data=all_shapes, ground_truth_name=label_name, representation_names=possible_features,
                                 k_eig=k_eig, use_cache=use_cache, save_to_cache=save_to_cache,
                                 cache_path=cached_data)

    test_loader = DataLoader(full_dataset, batch_size=None)
    model = diffusion_net.layers.DiffusionNet(C_in=c_in,
                                              C_out=n_class,
                                              C_width=width,
                                              N_block=depth,
                                              last_activation=lambda x: torch.nn.functional.log_softmax(x, dim=-1),
                                              outputs_at='vertices',
                                              dropout=True)
    model = model.to(device)
    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))

    worst_acc = 1.0
    best_acc = 0
    all_accs = []
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            features, verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels = data
            k_reps, k1k2_reps, hks_reps, shot_reps = features
            verts = verts.to(device)
            faces = faces.to(device)
            mass = mass.to(device)
            L = L.to(device)
            evals = evals.to(device)
            evecs = evecs.to(device)
            gradX = gradX.to(device)
            gradY = gradY.to(device)
            labels = labels.to(device)

            if input_features == 'xyz':
                feats = verts
                if random_rotate:
                    feats = diffusion_net.utils.random_rotate_points(feats)
            else:
                if input_features == 'k':
                    features = k_reps
                elif input_features == 'k1k2':
                    features = k1k2_reps
                elif input_features == 'hks':
                    features = hks_reps
                elif input_features == 'shot':
                    features = shot_reps
                elif input_features == 'shot16':
                    features = shot_reps
                    features = features.reshape((-1, 4, 4, 2, 2))[:, :2, :2, :, :].reshape((-1, 16))
                    features = torch.tensor(np.ascontiguousarray(features)).float()

                if len(features.shape) == 1:
                    features = features[:, None]
                feats = features
            feats = feats.to(device)

            preds = model(feats, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
            pred_labels = torch.argmax(preds, dim=-1)
            correct = pred_labels.eq(labels).sum().item()
            acc = correct / labels.shape[0]
            if acc > best_acc:
                best_acc = acc
            if acc < worst_acc:
                worst_acc = acc
                worst_verts = verts
                worst_faces = faces
                worst_labels = labels
                worst_preds = pred_labels
            all_accs.append(acc)
    mean_acc = np.mean(all_accs)
    return worst_verts, worst_faces, worst_acc, worst_labels, worst_preds, best_acc, mean_acc


def main(base_fold, dataset):
    data_fold = join(base_fold, dataset)
    data = join(data_fold, 'shapes')
    info = pd.read_csv(join(data_fold, 'info.csv'), index_col='idx', delimiter=';')
    cache = join(data_fold, 'cached_data.pt')
    meshes = []
    for rep in ['k', 'k1k2', 'xyz', 'hks', 'shot', 'shot16']:

        weights = join(data_fold, f'results/diffusion_net/{rep}/best_model.pt')
        v, f, acc, labs, preds, best_acc, mean_acc = run_segmentation_model(data_dir=data, info_df=info,
                                                                            weight_path=weights, label_name='labels',
                                                                            random_rotate=True, cached_data=cache,
                                                                            n_class=8, input_features=rep)

        err = labs - preds
        err[err != 0] = 1
        meshes.append([rep, v.numpy(), f.numpy(), err.numpy(), labs.numpy(), preds.numpy()])
        print(rep)
        print(acc)
        print(best_acc)
        print(mean_acc)

    ps.init()
    for m in meshes:
        rep, v, f, err, labs, preds = m
        ps_mesh = ps.register_surface_mesh(rep, v, f)
        ps_mesh.add_scalar_quantity(f"{rep}err", err, defined_on='vertices')
        ps_mesh.add_scalar_quantity(f"{rep}labs", labs, defined_on='vertices')
        ps_mesh.add_scalar_quantity(f"{rep}preds", preds, defined_on='vertices')
    ps.show()


if __name__ == '__main__':
    base_fold = 'Datasets/Shapes'
    main(base_fold, 'Human_pose')  # RNA,
