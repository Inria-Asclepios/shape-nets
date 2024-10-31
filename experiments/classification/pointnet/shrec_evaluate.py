import argparse
import logging
import os
from os.path import join, exists
import numpy as np
import pyvista as pv
import pandas as pd
import torch

from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torchmetrics.wrappers import MetricTracker

from shape_utils import ShapeDataset, metric_collection
from shapecentral.networks import point_net


def run_classification_model(data_dir, info_df, res_dir, label_name='labels',
                             n_fold=5, split_name_base='split_',
                             n_class=6, n_points=2048):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%d/%m/%Y %H:%M:%S')

    device = torch.device('cpu')
    logging.info(f'Using device {device}')

    input_features = 'xyz'
    c_in = 3

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

    full_dataset = ShapeDataset(data_list=all_shapes, labels=labels, input_name=input_features,
                                n_points=n_points, normalise_verts=True)

    for i in range(n_fold):
        res_folder = res_folders[i]
        base_acc = 0
        test_subset = Subset(full_dataset, n_fold_test_ids[i])
        test_loader = DataLoader(test_subset)

        # model and optimizer
        model = point_net.PointNetClassification(in_channels=c_in, n_classes=n_class)
        model = model.to(device)

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

            test_acc = tracker_test.compute()['acc'].item()
            if test_acc > base_acc:
                torch.save(model.state_dict(), join(res_folder, 'best_model.pt'))
                base_acc = test_acc

        logging.info(f"Test state: {tracker_test.compute()}")
        torch.save(tracker_test.compute_all(), join(res_folder, 'all_test_metrics.pt'))


def main(base_fold):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%d/%m/%Y %H:%M:%S')

    logging.info('Starting Shrec_16_coarse experiment')
    data_fold = join(base_fold, 'Shrec_16_coarse')
    data = join(data_fold, 'shapes')
    info = pd.read_csv(join(data_fold, 'info.csv'), index_col='idx', delimiter=';')
    res_fold = join(data_fold, 'results', 'point_net')
    if not exists(res_fold):
        os.mkdir(res_fold)
    split_name = '10_split_'

    res = join(res_fold, 'xyz')
    run_classification_model(data_dir=data, info_df=info, res_dir=res, label_name='labels',
                             n_fold=5, split_name_base=split_name, n_class=30)
    logging.info('Shrec_16_coarse experiment is done')


if __name__ == '__main__':
    base_fold = 'Datasets/Shapes'
    main(base_fold)
