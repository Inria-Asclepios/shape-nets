import numpy as np
import pyvista as pv
import torch
from os import listdir
from os.path import join
from natsort import natsorted
from tqdm import tqdm
import igl
from shapecentral.networks import diffusion_net


def compute_reps(mesh):
    verts = mesh.points
    faces = mesh.faces.reshape(-1, 4)[:, 1:]

    _, _, k1, k2 = igl.principal_curvature(verts, faces)
    k = k1*k2

    verts = torch.tensor(np.ascontiguousarray(verts)).float()
    faces = torch.tensor(np.ascontiguousarray(faces))
    _, _, _, evals, evecs, _, _ = diffusion_net.geometry.get_operators(verts, faces, k_eig=128)
    hks = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, count=16)

    mesh['hks'] = hks.numpy()
    mesh['k'] = k
    mesh['k1k2'] = np.stack((k1, k2), axis=-1)

    return mesh

def main():
    data_folder = 'Datasets/Shapes/Human_pose/shapes'
    mesh_paths = natsorted([join(data_folder, el) for el in listdir(data_folder)])

    res_folder = 'Datasets/Shapes/Human_pose/shapes_noise_7'
    for i in tqdm(range(len(mesh_paths))):
        mesh = pv.read(mesh_paths[i])
        noise = np.random.normal(0, mesh.length*0.007, size=mesh.points.shape)
        new_coords = mesh.points + noise

        new_mesh = mesh.copy()
        new_mesh.clear_data()
        new_mesh.points = new_coords
        new_mesh = compute_reps(new_mesh)
        new_mesh.point_data['labels'] = mesh.point_data['labels']
        new_mesh.save(join(res_folder, mesh_paths[i].split('/')[-1]), binary=False)


if __name__ == '__main__':
    main()
