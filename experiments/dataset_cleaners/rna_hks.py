from os import listdir
from os.path import join
from tqdm import tqdm

import torch
import numpy as np
import pyvista as pv
from shapecentral.networks import diffusion_net

def main():
    rna_folder = 'Datasets/Shapes/RNA/shapes'
    all_names = [el for el in listdir(rna_folder) if not el.startswith('.')]
    all_meshes = [pv.read(join(rna_folder, el)) for el in all_names]
    all_verts = [torch.tensor(np.ascontiguousarray(el.points)).float() for el in all_meshes]
    all_faces = [torch.tensor(np.ascontiguousarray(el.faces.reshape(-1, 4)[:, 1:])) for el in all_meshes]

    _, _, _, all_evals, all_evecs, _, _ = diffusion_net.geometry.get_all_operators(all_verts, all_faces, k_eig=128)

    for i, name in tqdm(enumerate(all_names)):
        mesh = all_meshes[i]
        hks = diffusion_net.geometry.compute_hks_autoscale(all_evals[i], all_evecs[i], count=16)
        mesh.point_data['hks'] = hks.numpy()
        mesh.save(join(rna_folder, name), binary=False)

    breakpoint()

if __name__ == '__main__':
    main()
