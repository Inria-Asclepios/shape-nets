import os
import numpy as np
import pyvista as pv
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import potpourri3d as pp3d
from torch_geometric.nn import fps
from tqdm import tqdm
import torchmetrics as tm
import torchmetrics.classification as tmc

from shapecentral.networks.diffusion_net.geometry import normalize_positions


def mesh_sampler(vertices, faces, features, labels, n_points=1024):
    ratio = n_points / len(vertices)
    sampled_idxs = fps(vertices, ratio=ratio)

    final_verts = vertices[sampled_idxs]
    final_features = features[sampled_idxs]
    final_labels = labels[sampled_idxs]

    return final_verts, final_features, final_labels


class ShapeDataset(Dataset):
    def __init__(self, data_list, ground_truth_name, input_name=None, n_points=None, normalise_verts=True):
        self.ground_truth_name = ground_truth_name
        self.input_name = input_name
        self.n_points = n_points
        self.normalise_verts = normalise_verts
        if type(data_list) is str:
            self.data_list = [pv.read(f) for f in os.listdir(data_list)]
        elif type(data_list) is list:
            self.data_list = data_list
        else:
            raise TypeError('You need to pass data either as list of meshes or directory')

        if self.input_name not in ['xyz', 'k1k2n', 'shot16']:
            all_names = self.data_list[0].array_names
            assert self.input_name in all_names, f"{self.input_name} is missing from representations"

        self.all_data = []
        self.min_label = np.min([np.min(s.point_data[self.ground_truth_name]) for s in self.data_list])
        for shape in tqdm(self.data_list):
            verts = shape.points
            faces = shape.faces.reshape(-1, 4)[:, 1:]
            labels = shape.point_data[self.ground_truth_name]
            labels = labels - self.min_label
            if self.input_name == 'xyz':
                reps = verts
                reps = torch.tensor(np.ascontiguousarray(reps)).float()
            elif self.input_name == 'shot16':
                reps = shape['shot']
                reps = reps.reshape((-1, 4, 4, 2, 2))[:, :2, :2, :, :].reshape((-1, 16))
                reps = torch.tensor(np.ascontiguousarray(reps)).float()
            elif self.input_name == 'k1k2n':
                reps = shape['k1k2']
                reps = torch.tensor(np.ascontiguousarray(reps)).float()
                reps = reps - torch.mean(reps)
            else:
                reps = shape[self.input_name]
                reps = torch.tensor(np.ascontiguousarray(reps)).float()

            verts = torch.tensor(np.ascontiguousarray(verts)).float()
            faces = torch.tensor(np.ascontiguousarray(faces))
            labels = torch.tensor(np.ascontiguousarray(labels)).long()

            if len(reps.shape) == 1:
                reps = reps[:, None]

            if self.normalise_verts:
                verts = normalize_positions(verts)
            if self.n_points is not None:
                verts, reps, labels = mesh_sampler(verts, faces, reps, labels, self.n_points)

            data = Data(x=reps, y=labels, pos=verts)
            self.all_data.append(data)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.all_data[idx]


def metric_collection(n_classes, device):
    t, n, mav, avw, avmi, avma = 'multiclass', n_classes, 'global', 'weighted', 'micro', 'macro'
    metrics_dict = {'acc': tmc.Accuracy(task=t, num_classes=n).to(device),
                    'precision_micro': tmc.Precision(task=t, num_classes=n, average=avmi).to(device),
                    'precision_macro': tmc.Precision(task=t, num_classes=n, average=avma).to(device),
                    'precision_weighted': tmc.Precision(task=t, num_classes=n, average=avw).to(device),
                    'recall_micro': tmc.Recall(task=t, num_classes=n, average=avmi).to(device),
                    'recall_macro': tmc.Recall(task=t, num_classes=n, average=avma).to(device),
                    'recall_weighted': tmc.Recall(task=t, num_classes=n, average=avw).to(device),
                    'specificity_micro': tmc.Specificity(task=t, num_classes=n, average=avmi).to(device),
                    'specificity_macro': tmc.Specificity(task=t, num_classes=n, average=avma).to(device),
                    'specificity_weighted': tmc.Specificity(task=t, num_classes=n, average=avw).to(device),
                    'dice_micro': tmc.Dice(num_classes=n, average=avmi, mdmc_average=mav).to(device),
                    'dice_macro': tmc.Dice(num_classes=n, average=avma, mdmc_average=mav).to(device),
                    'f1_micro': tmc.MulticlassF1Score(num_classes=n, average=avmi, multidim_average=mav).to(device),
                    'f1_macro': tmc.MulticlassF1Score(num_classes=n, average=avma, multidim_average=mav).to(device),
                    'f1_weighted': tmc.MulticlassF1Score(num_classes=n, average=avw, multidim_average=mav).to(device)}

    return tm.MetricCollection(metrics_dict)
