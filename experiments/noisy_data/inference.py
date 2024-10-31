from os.path import join, exists
import pyvista as pv
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from shapecentral.networks import diffusion_net
from shape_utils import shape_dataset

def infer_diffnet(data_dir, info_df, weight_path, label_name='labels', random_rotate=True,
                           n_class=6, input_features='k', width=128, depth=4, k_eig=128, cached_data=None):

    device = torch.device('cpu')

    possible_features = ['k', 'k1k2', 'xyz', 'hks', 'kn', 'k1k2n']
    if input_features in ['k', 'kn']:
        c_in = 1
    elif input_features in ['k1k2', 'k1k2n']:
        c_in = 2
    elif input_features == 'xyz':
        c_in = 3
    elif input_features == 'hks':
        c_in = 16
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

    all_accs = []
    model.eval()

    with torch.no_grad():
        for data in test_loader:
            features, verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels = data
            k_reps, k1k2_reps, hks_reps, kn_reps, k1k2n_reps, = features
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
                elif input_features == 'kn':
                    features = kn_reps
                elif input_features == 'k1k2n':
                    features = k1k2n_reps

                if len(features.shape) == 1:
                    features = features[:, None]
                feats = features
            feats = feats.to(device)

            preds = model(feats, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
            pred_labels = torch.argmax(preds, dim=-1)
            correct = pred_labels.eq(labels).sum().item()
            acc = correct / labels.shape[0]
            all_accs.append(acc)
    return all_accs


def main(base_fold, dataset):
    data_fold = join(base_fold, dataset)

    print('Starting baseline')
    d0 = {k: [] for k in ['k1k2', 'xyz', 'hks']}
    data = join(data_fold, 'shapes_noise_0')
    cache = join(data_fold, 'cached_data_0.pt')
    info = pd.read_csv(join(data_fold, 'info.csv'), index_col='idx', delimiter=';')
    for rep in ['k1k2', 'xyz', 'hks']:
        weights = join(data_fold, f'results/diffusion_net/{rep}/run_5/best_model.pt')
        accs = infer_diffnet(data_dir=data, info_df=info, weight_path=weights, label_name='labels',
                             random_rotate=True, cached_data=cache, n_class=8, input_features=rep)
        print(f"Mean accuracy on the full dataset for input {rep} is {np.mean(accs)}")
        d0[rep] = accs

    print('####################################################################################')
    print('Starting 1% Noise')
    d1 = {k: [] for k in ['k1k2', 'xyz', 'hks']}
    data = join(data_fold, 'shapes_noise_1')
    cache = join(data_fold, 'cached_data_1.pt')
    info = pd.read_csv(join(data_fold, 'info.csv'), index_col='idx', delimiter=';')
    for rep in ['k1k2', 'xyz', 'hks']:
        weights = join(data_fold, f'results/diffusion_net/{rep}/run_5/best_model.pt')
        accs = infer_diffnet(data_dir=data, info_df=info, weight_path=weights, label_name='labels',
                             random_rotate=True, cached_data=cache, n_class=8, input_features=rep)
        print(f"Mean accuracy on the full dataset for input {rep} is {np.mean(accs)}")
        d1[rep] = accs

    print('####################################################################################')
    print('Starting 3% Noise')
    d3 = {k: [] for k in ['k1k2', 'xyz', 'hks']}
    data = join(data_fold, 'shapes_noise_3')
    cache = join(data_fold, 'cached_data_3.pt')
    info = pd.read_csv(join(data_fold, 'info.csv'), index_col='idx', delimiter=';')
    for rep in ['k1k2', 'xyz', 'hks']:
        weights = join(data_fold, f'results/diffusion_net/{rep}/run_5/best_model.pt')
        accs = infer_diffnet(data_dir=data, info_df=info, weight_path=weights, label_name='labels',
                             random_rotate=True, cached_data=cache, n_class=8, input_features=rep)
        print(f"Mean accuracy on the full dataset for input {rep} is {np.mean(accs)}")
        d3[rep] = accs

    print('####################################################################################')
    print('Starting 5% Noise')
    d5 = {k: [] for k in ['k1k2', 'xyz', 'hks']}
    data = join(data_fold, 'shapes_noise_5')
    cache = join(data_fold, 'cached_data_5.pt')
    info = pd.read_csv(join(data_fold, 'info.csv'), index_col='idx', delimiter=';')
    for rep in ['k1k2', 'xyz', 'hks']:
        weights = join(data_fold, f'results/diffusion_net/{rep}/run_5/best_model.pt')
        accs = infer_diffnet(data_dir=data, info_df=info, weight_path=weights, label_name='labels',
                             random_rotate=True, cached_data=cache, n_class=8, input_features=rep)
        print(f"Mean accuracy on the full dataset for input {rep} is {np.mean(accs)}")
        d5[rep] = accs

    print('####################################################################################')
    print('Starting 7% Noise')
    d7 = {k: [] for k in ['k1k2', 'xyz', 'hks']}
    data = join(data_fold, 'shapes_noise_7')
    cache = join(data_fold, 'cached_data_7.pt')
    info = pd.read_csv(join(data_fold, 'info.csv'), index_col='idx', delimiter=';')
    for rep in ['k1k2', 'xyz', 'hks']:
        weights = join(data_fold, f'results/diffusion_net/{rep}/run_5/best_model.pt')
        accs = infer_diffnet(data_dir=data, info_df=info, weight_path=weights, label_name='labels',
                             random_rotate=True, cached_data=cache, n_class=8, input_features=rep)
        print(f"Mean accuracy on the full dataset for input {rep} is {np.mean(accs)}")
        d7[rep] = accs

    print('####################################################################################')
    print('Starting 10% Noise')
    d10 = {k: [] for k in ['k1k2', 'xyz', 'hks']}
    data = join(data_fold, 'shapes_noise_10')
    cache = join(data_fold, 'cached_data_10.pt')
    info = pd.read_csv(join(data_fold, 'info.csv'), index_col='idx', delimiter=';')
    for rep in ['k1k2', 'xyz', 'hks']:
        weights = join(data_fold, f'results/diffusion_net/{rep}/run_5/best_model.pt')
        accs = infer_diffnet(data_dir=data, info_df=info, weight_path=weights, label_name='labels',
                             random_rotate=True, cached_data=cache, n_class=8, input_features=rep)
        print(f"Mean accuracy on the full dataset for input {rep} is {np.mean(accs)}")
        d10[rep] = accs

    d0 = pd.DataFrame(d0)
    d1 = pd.DataFrame(d1)
    d3 = pd.DataFrame(d3)
    d5 = pd.DataFrame(d5)
    d7 = pd.DataFrame(d7)
    d10 = pd.DataFrame(d10)

    final_res = pd.concat([d0, d1, d3, d5, d7, d10], keys=[0, 1, 3, 5, 7, 10], axis=1)
    final_res.to_csv(join(data_fold, 'results.csv'))

if __name__ == '__main__':
    base_fold = 'Datasets/Shapes'
    main(base_fold, 'Human_pose')
