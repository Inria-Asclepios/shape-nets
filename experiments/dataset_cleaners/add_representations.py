from os import listdir, mkdir
from os.path import join
from tqdm import tqdm
import numpy as np
import pyvista as pv
from shapecentral import Surface


def main():
    rna_base = 'Datasets/Shapes/RNA/shapesv0'
    humans_base = 'Datasets/Shapes/Human_pose/shapesv0'
    shrec_coarse_base = 'Datasets/Shapes/Shrec_16/shapes_coarsev0'

    print('Starting RNA')
    rna_res = rna_base[:rna_base.rfind("v0")]
    mkdir(rna_res)
    names = [p for p in listdir(rna_base) if not p.startswith('.')]
    for name in tqdm(names):
        if name in listdir(rna_res):
            print(f"{name.split('.')[0]} is done")
        else:
            print(f"Doing {name.split('.')[0]}")  # , {i} / {len(names)}")
            mesh_path = join(rna_base, name)
            mesh = pv.read(mesh_path)
            print(f'n verts before: {len(mesh.points)}')
            mesh = mesh.clean()  # Because of 488
            print(f'n verts after: {len(mesh.points)}')
            labels = mesh['labels']
            mesh.clear_data()
            surf = Surface(shape=mesh)
            surf.compute_curvature_representers()
            surf.compute_signature()
            surf.compute_shot_descriptors(radius=15, verbose=False)
            final_mesh = surf.as_polydata()
            final_mesh.point_data['labels'] = labels
            final_mesh.point_data['k1k2'] = np.stack([final_mesh['kmin'], final_mesh['kmax']], axis=1)
            final_mesh.save(join(rna_res, name))
    print('RNA Done')

    print('Starting humans')
    humans_res = humans_base[:humans_base.rfind("v0")]
    mkdir(humans_res)
    names = [p for p in listdir(humans_base) if not p.startswith('.')]
    for name in tqdm(names):
        if name in listdir(humans_res):
            print(f"{name.split('.')[0]} is done")
        else:
            print(f" {name.split('.')[0]}")
            mesh_path = join(humans_base, name)
            mesh = pv.read(mesh_path)
            labels = mesh['labels']
            mesh.clear_data()
            surf = Surface(shape=mesh)
            surf.compute_curvature_representers()
            surf.compute_signature()
            surf.compute_shot_descriptors(radius=15, verbose=False)
            final_mesh = surf.as_polydata()
            final_mesh.point_data['labels'] = labels
            final_mesh.point_data['k1k2'] = np.stack([final_mesh['kmin'], final_mesh['kmax']], axis=1)
            final_mesh.save(join(humans_res, name))
    print('Humans done')

    print('Starting Shrec coarse')
    shrec_coarse_res = shrec_coarse_base[:shrec_coarse_base.rfind("v0")]
    mkdir(shrec_coarse_res)
    names = [p for p in listdir(shrec_coarse_base) if not p.startswith('.')]
    for name in tqdm(names):
        if name in listdir(shrec_coarse_res):
            print(f"{name.split('.')[0]} is done")
        else:
            print(f" {name.split('.')[0]}")
            mesh_path = join(shrec_coarse_base, name)
            mesh = pv.read(mesh_path)
            mesh.clear_data()
            surf = Surface(shape=mesh)
            surf.compute_curvature_representers()
            surf.compute_signature()
            surf.compute_shot_descriptors(radius=25, verbose=False)
            final_mesh = surf.as_polydata()
            final_mesh.point_data['k1k2'] = np.stack([final_mesh['kmin'], final_mesh['kmax']], axis=1)
            final_mesh.save(join(shrec_coarse_res, name))
    print('Shrec coarse done')

    breakpoint()


if __name__ == '__main__':
    main()
