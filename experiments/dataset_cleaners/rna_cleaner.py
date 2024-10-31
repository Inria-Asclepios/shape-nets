import numpy as np
import pyvista as pv
import pandas as pd
from os import listdir
from os.path import join
from natsort import natsorted
from cleaner_tools import attach_reps, shuffle_split
from tqdm import tqdm


def main():
    info = pd.DataFrame({'name': [], 'train_split': []}, index=[])
    res_dir = 'Datasets/shape_datasets/RNA/shapes'
    test_txt = 'Datasets/shape_datasets/RNA/test.txt'
    rna_dataset = 'Datasets/shape_datasets/RNA/original/off'
    mesh_paths = natsorted([join(rna_dataset, el) for el in listdir(rna_dataset)])
    rna_dataset_seg = 'Datasets/shape_datasets/RNA/original/labels'

    with open(test_txt) as f:
        lines = f.readlines()
    test_name = []
    for l in lines:
        name = l.split('.')[0]
        test_name.append(name)

    c = 0
    for i in range(len(mesh_paths)):
        name = mesh_paths[i].split('/')[-1].split('.')[0]
        if name in test_name:
            train_status = 1
        else:
            train_status = 0
        info.loc[c] = [name, train_status]

        seg_path = join(rna_dataset_seg, name+'.txt')
        mesh = pv.read(mesh_paths[i])
        mesh.point_data['labels'] = np.loadtxt(seg_path)
        mesh = mesh.extract_surface()

        mesh.save(join(res_dir, str(c)+'.vtk'), binary=False)
        c += 1

        if i % 60 == 0:
            print(i)

    info.to_csv('Datasets/shape_datasets/RNA/info.csv')

    info = pd.read_csv('Datasets/Shapes/RNA/info.csv')
    shape_path = 'Datasets/shape_datasets/RNA/shapes'
    res_path = 'Datasets/Shapes/RNA/shapes'
    all_shapes = [el for el in listdir(shape_path) if not el.startswith('.')]

    for p in tqdm(all_shapes):
        shape = pv.read(join(shape_path, p))
        new_shape = shape.copy()
        new_shape.clear_data()
        new_shape = attach_reps(new_shape)
        new_shape.point_data['labels'] = shape.point_data['segmentation_gt']
        new_shape.save(join(res_path, p), binary=False)

    print('creating train splits')

    info = shuffle_split(info, name='split')

    info.to_csv('Datasets/Shapes/RNA/info.csv')

    print('done')

    breakpoint()


if __name__ == '__main__':
    main()
