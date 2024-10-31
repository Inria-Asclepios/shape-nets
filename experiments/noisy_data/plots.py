import pyvista as pv
from os import listdir
from os.path import join
from natsort import natsorted
from tqdm import tqdm
import numpy as np



def main():
    noise_1 = 'Datasets/Shapes/Human_pose/shapes_noise_1'
    noise_5 = 'Datasets/Shapes/Human_pose/shapes_noise_5'
    noise_10 = 'Datasets/Shapes/Human_pose/shapes_noise_10'
    noise_3 = 'Datasets/Shapes/Human_pose/shapes_noise_3'
    noise_7 = 'Datasets/Shapes/Human_pose/shapes_noise_7'
    mesh_ids = natsorted(listdir(noise_1))

    mesh = pv.read('Datasets/cheburashka.off')
    noise = np.random.normal(0, mesh.length * 0.001, size=mesh.points.shape)
    new_coords = mesh.points + noise
    new_mesh_1 = mesh.copy()
    new_mesh_1.points = new_coords

    noise = np.random.normal(0, mesh.length * 0.003, size=mesh.points.shape)
    new_coords = mesh.points + noise
    new_mesh_3 = mesh.copy()
    new_mesh_3.points = new_coords

    noise = np.random.normal(0, mesh.length * 0.005, size=mesh.points.shape)
    new_coords = mesh.points + noise
    new_mesh_5 = mesh.copy()
    new_mesh_5.points = new_coords

    noise = np.random.normal(0, mesh.length * 0.007, size=mesh.points.shape)
    new_coords = mesh.points + noise
    new_mesh_7 = mesh.copy()
    new_mesh_7.points = new_coords

    noise = np.random.normal(0, mesh.length * 0.010, size=mesh.points.shape)
    new_coords = mesh.points + noise
    new_mesh_10 = mesh.copy()
    new_mesh_10.points = new_coords

    p = pv.Plotter(shape=(2, 3))
    p.subplot(0, 0)
    p.add_mesh(mesh)
    p.subplot(0, 1)
    p.add_mesh(new_mesh_1)
    p.subplot(0, 2)
    p.add_mesh(new_mesh_3)
    p.subplot(1, 0)
    p.add_mesh(new_mesh_5)
    p.subplot(1, 1)
    p.add_mesh(new_mesh_7)
    p.subplot(1, 2)
    p.add_mesh(new_mesh_10)
    p.link_views()
    p.show()

    for idx in tqdm(range(len(mesh_ids))):
        if idx % 2 == 0:
            mesh_1 = pv.read(join(noise_1, mesh_ids[idx]))
            mesh_5 = pv.read(join(noise_5, mesh_ids[idx]))
            mesh_10 = pv.read(join(noise_10, mesh_ids[idx]))
            mesh_3 = pv.read(join(noise_3, mesh_ids[idx]))
            mesh_7 = pv.read(join(noise_7, mesh_ids[idx]))

            p = pv.Plotter(shape=(1, 5))
            p.subplot(0, 0)
            p.add_mesh(mesh_1, scalars='labels')
            p.subplot(0, 1)
            p.add_mesh(mesh_3, scalars='labels')
            p.subplot(0, 2)
            p.add_mesh(mesh_5, scalars='labels')
            p.subplot(0, 3)
            p.add_mesh(mesh_7, scalars='labels')
            p.subplot(0, 4)
            p.add_mesh(mesh_10, scalars='labels')

            p.link_views()
            p.show()



if __name__ == '__main__':
    main()
