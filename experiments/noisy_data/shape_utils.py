import os
from tqdm import tqdm
import numpy as np
import pyvista as pv
import torch
from torch.utils.data import Dataset
from shapecentral.networks import diffusion_net

import torchmetrics as tm
import torchmetrics.classification as tmc


class shape_dataset(Dataset):
    def __init__(self, data, ground_truth_name, representation_names=None, k_eig=128, normalise_rep=False,
                 use_cache=False, save_to_cache=False, cache_path=None):
        if isinstance(representation_names, str):
            self.representation_names = [representation_names]
        else:
            self.representation_names = representation_names
        self.ground_truth_name = ground_truth_name
        self.k_eig = k_eig
        self.normalise_rep = normalise_rep

        self.use_cache = use_cache
        self.save_to_cache = save_to_cache
        self.cache = cache_path

        if type(data) is str:
            self.data = [pv.read(f) for f in os.listdir(data)]
        elif type(data) is list:
            self.data = data
        else:
            raise TypeError('You need to pass data either as list of meshes or directory')

        if self.use_cache:
            print('loading from cache')
            cached_info = torch.load(self.cache)
            self.all_reps, self.all_labels, self.all_verts, self.all_faces = cached_info[0:4]
            self.frames, self.massvec, self.L, self.evals, self.evecs, self.gradX, self.gradY = cached_info[4:]
            print('cache loaded')

        else:
            self.min_label = np.min([np.min(s.point_data[self.ground_truth_name]) for s in self.data])
            self.all_verts = []
            self.all_faces = []
            self.all_labels = []

            self.all_reps = []
            N = len(self.data)
            n = 0
            for shape in tqdm(self.data):
                n += 1
                verts = shape.points
                faces = shape.faces.reshape(-1, 4)[:, 1:]
                labels = shape.point_data[self.ground_truth_name]
                labels = labels - self.min_label

                k_reps = None
                k1k2_reps = None
                kn_reps = None
                k1k2n_reps = None
                hks_reps = None
                for r in self.representation_names:
                    if r == 'k':
                        k_reps = shape.point_data[r]
                        k_reps = torch.tensor(np.ascontiguousarray(k_reps)).float()
                    if r == 'k1k2':
                        k1k2_reps = shape.point_data[r]
                        k1k2_reps = torch.tensor(np.ascontiguousarray(k1k2_reps)).float()
                    if r == 'hks':
                        hks_reps = shape.point_data[r]
                        hks_reps = torch.tensor(np.ascontiguousarray(hks_reps)).float()
                    if r == 'kn':
                        kn_reps = shape.point_data[r]
                        kn_reps = torch.tensor(np.ascontiguousarray(kn_reps)).float()
                    if r == 'k1k2n':
                        k1k2n_reps = shape.point_data[r]
                        k1k2n_reps = torch.tensor(np.ascontiguousarray(k1k2n_reps)).float()

                verts = torch.tensor(np.ascontiguousarray(verts)).float()
                verts = diffusion_net.geometry.normalize_positions(verts)
                faces = torch.tensor(np.ascontiguousarray(faces))
                labels = torch.tensor(np.ascontiguousarray(labels)).long()

                self.all_verts.append(verts)
                self.all_faces.append(faces)
                self.all_labels.append(labels)
                self.all_reps.append([k_reps, k1k2_reps, hks_reps, kn_reps, k1k2n_reps])

            self.operator_vals = diffusion_net.geometry.get_all_operators(self.all_verts, self.all_faces, self.k_eig)
            self.frames, self.massvec, self.L, self.evals, self.evecs, self.gradX, self.gradY = self.operator_vals

        if self.save_to_cache:
            print('saved to cache')
            torch.save((self.all_reps, self.all_labels, self.all_verts, self.all_faces, self.frames, self.massvec,
                        self.L, self.evals, self.evecs, self.gradX, self.gradY), self.cache)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.all_reps[idx], self.all_verts[idx], self.all_faces[idx], self.frames[idx], self.massvec[idx],
                self.L[idx], self.evals[idx], self.evecs[idx], self.gradX[idx],
                self.gradY[idx], self.all_labels[idx])


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
