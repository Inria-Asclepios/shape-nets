import logging
import os
from os.path import join, exists
import numpy as np
import pyvista as pv
import pandas as pd
import torch

from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from torchmetrics.wrappers import MetricTracker

from shape_utils import ShapeDataset, metric_collection
from shapecentral.networks import delta_conv

def run_classification_model(data_dir, info_df, res_dir, weights_path, label_name='labels',
                             n_fold=5, split_name_base='split_',
                             n_class=6, width=128, depth=4, n_points=2048,
                             num_neighbors=20, grad_regularizer=0.001, grad_kernel_width=1):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%d/%m/%Y %H:%M:%S')

    device = torch.device('cpu')
    logging.info(f'Using device {device}')

    input_features = 'xyz'
    c_in = 3

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

    n_fold_test_ids = [np.argwhere(info_df[splits[i]] == 0).flatten() for i in range(n_fold)]

    labels = info_df[label_name].to_list()
    all_shapes = [pv.read(join(data_dir, str(i) + '.vtk')) for i in info_df.index]

    pre_transform = Compose([delta_conv.transforms.NormalizeScale()])

    full_dataset = ShapeDataset(data_list=all_shapes, labels=labels, input_name=input_features,
                                pre_transform=pre_transform, n_points=n_points)

    if isinstance(width, list):
        channels = width
    else:
        channels = [width] * depth

    for i in range(n_fold):
        res_folder = res_folders[i]

        test_subset = Subset(full_dataset, n_fold_test_ids[i])
        test_loader = DataLoader(test_subset, shuffle=True, batch_size=16)

        # model
        model = delta_conv.models.DeltaNetClassification(in_channels=c_in,
                                                         num_classes=n_class,
                                                         conv_channels=channels,
                                                         num_neighbors=num_neighbors,
                                                         grad_regularizer=grad_regularizer,
                                                         grad_kernel_width=grad_kernel_width)
        model = model.to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))

        metrics_test = metric_collection(n_classes=n_class, device=device)
        tracker_test = MetricTracker(metrics_test)
        tracker_test.increment()
        model.eval()
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                preds = model(data)
                labels = data.y.long()
                pred_labels = torch.argmax(preds, dim=-1)
                tracker_test.update(pred_labels, labels)

            logging.info(f"Test state: {tracker_test.compute()}")

    metric_test_results_all = tracker_test.compute_all()
    torch.save(metric_test_results_all, join(res_folder, 'all_test_metrics.pt'))


def main(base_fold):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%d/%m/%Y %H:%M:%S')
    logging.info('Starting Shrec_16 experiment')
    data_fold = join(base_fold, 'Shrec_16')
    data = join(data_fold, 'shapes')
    info = pd.read_csv(join(data_fold, 'info.csv'), index_col='idx', delimiter=';')
    res_fold = join(data_fold, 'results', 'delta_conv')
    weights_path = join(data_fold, 'eval_weights/delta.pt')
    if not exists(res_fold):
        os.mkdir(res_fold)
    split_name = '10_split_'
    n = 30
    n_pts = 2048

    res = join(res_fold, 'xyz_eval')
    run_classification_model(data_dir=data, info_df=info, res_dir=res, label_name='labels',
                             n_fold=5, split_name_base=split_name, n_points=n_pts, weights_path=weights_path,
                             n_class=n, width=32, depth=4,
                             num_neighbors=20, grad_regularizer=0.001, grad_kernel_width=1)

    logging.info('Shrec_16 experiment is done')


if __name__ == '__main__':
    base_fold = 'Datasets/Shapes'
    main(base_fold)
