import numpy as np
import pyvista as pv
import pandas as pd
from os import listdir
from os.path import join
from natsort import natsorted
from cleaner_tools import attach_reps, shuffle_split

def main():
    info = pd.DataFrame({'train/test': [], 'dataset': [], 'name': []}, index=[])
    res_dir = 'Datasets/Shapes/Human_pose/shapes'

    c = 0

    shrec_dataset = 'Datasets/sig17_seg_benchmark/original/meshes/test/shrec'
    mesh_paths = natsorted([join(shrec_dataset, el) for el in listdir(shrec_dataset)])
    shrec_dataset_seg = 'Datasets/sig17_seg_benchmark/original/segs/test/shrec'
    seg_paths = natsorted([join(shrec_dataset_seg, el) for el in listdir(shrec_dataset_seg)])
    dataset_name = 'shrec'
    train_status = '1'  # meaning testing dataset
    for i in range(len(mesh_paths)):
        name = mesh_paths[i].split('/')[-1].split('.')[0]
        info.loc[c] = [train_status, dataset_name, name]

        mesh = pv.read(mesh_paths[i])
        mesh = mesh.extract_surface()
        mesh.cell_data['segmentation_gt'] = np.loadtxt(seg_paths[i])

        mesh = mesh.cell_data_to_point_data()
        mesh.cell_data['original_segmentation_gt'] = np.loadtxt(seg_paths[i])
        mesh.point_data['segmentation_gt'] = np.round(mesh.point_data['segmentation_gt'])
        mesh.save(join(res_dir, str(c) + '.vtk'), binary=False)
        c += 1

    adobe_dataset = 'Datasets/sig17_seg_benchmark/original/meshes/train/adobe'
    mesh_paths = natsorted([join(adobe_dataset, el) for el in listdir(adobe_dataset)])
    adobe_dataset_seg = 'Datasets/sig17_seg_benchmark/original/segs/train/adobe'
    dataset_name = 'adobe'
    train_status = '0'  # meaning training dataset
    for i in range(len(mesh_paths)):
        name = mesh_paths[i].split('/')[-1].split('.')[0]
        info.loc[c] = [train_status, dataset_name, name]

        seg_path = join(adobe_dataset_seg, name+'.txt')
        mesh = pv.read(mesh_paths[i])
        mesh = mesh.extract_surface()
        mesh.cell_data['segmentation_gt'] = np.loadtxt(seg_path)
        mesh = mesh.cell_data_to_point_data()
        mesh.cell_data['original_segmentation_gt'] = np.loadtxt(seg_path)
        mesh.point_data['segmentation_gt'] = np.round(mesh.point_data['segmentation_gt'])
        mesh.save(join(res_dir, str(c) + '.vtk'), binary=False)

        c += 1

    faust_dataset = 'Datasets/sig17_seg_benchmark/original/meshes/train/faust'
    mesh_paths = natsorted([join(faust_dataset, el) for el in listdir(faust_dataset)])
    faust_seg = 'Datasets/sig17_seg_benchmark/original/segs/train/faust/faust_corrected.txt'
    dataset_name = 'faust'
    train_status = '0'  # meaning training dataset
    for i in range(len(mesh_paths)):
        name = mesh_paths[i].split('/')[-1].split('.')[0]
        info.loc[c] = [train_status, dataset_name, name]

        mesh = pv.read(mesh_paths[i])
        mesh = mesh.extract_surface()
        mesh.cell_data['segmentation_gt'] = np.loadtxt(faust_seg)
        mesh = mesh.cell_data_to_point_data()
        mesh.cell_data['original_segmentation_gt'] = np.loadtxt(faust_seg)
        mesh.point_data['segmentation_gt'] = np.round(mesh.point_data['segmentation_gt'])
        mesh.save(join(res_dir, str(c) + '.vtk'), binary=False)
        c += 1

    scape_dataset = 'Datasets/sig17_seg_benchmark/original/meshes/train/scape'
    mesh_paths = natsorted([join(scape_dataset, el) for el in listdir(scape_dataset)])
    scape_seg = 'Datasets/sig17_seg_benchmark/original/segs/train/scape/scape_corrected.txt'
    dataset_name = 'scape'
    train_status = '0'  # meaning training dataset
    for i in range(len(mesh_paths)):
        name = mesh_paths[i].split('/')[-1].split('.')[0]
        info.loc[c] = [train_status, dataset_name, name]

        mesh = pv.read(mesh_paths[i])
        mesh = mesh.extract_surface()
        mesh.cell_data['segmentation_gt'] = np.loadtxt(scape_seg)
        mesh = mesh.cell_data_to_point_data()
        mesh.cell_data['original_segmentation_gt'] = np.loadtxt(scape_seg)
        mesh.point_data['segmentation_gt'] = np.round(mesh.point_data['segmentation_gt'])
        mesh.save(join(res_dir, str(c) + '.vtk'), binary=False)
        c += 1

    mit_dataset = 'Datasets/sig17_seg_benchmark/original/meshes/train/mit'
    seg_dataset = 'Datasets/sig17_seg_benchmark/original/segs/train/mit'
    dataset_name = 'mit'
    train_status = '0'  # meaning training dataset
    for el in listdir(mit_dataset):
        name_1 = el.split('_')[-1]
        name_seg = join(seg_dataset, 'mit_'+name_1+'_corrected.txt')
        sub_dataset = join(mit_dataset, el, 'meshes')
        mesh_paths = natsorted([join(sub_dataset, el) for el in listdir(sub_dataset)])
        for i in range(len(mesh_paths)):
            name_2 = mesh_paths[i].split('/')[-1].split('.')[0].split('_')[-1]
            name = name_1 + '_' + name_2
            info.loc[c] = [train_status, dataset_name, name]

            mesh = pv.read(mesh_paths[i])
            mesh = mesh.extract_surface()
            mesh.cell_data['segmentation_gt'] = np.loadtxt(name_seg)
            mesh = mesh.cell_data_to_point_data()
            mesh.cell_data['original_segmentation_gt'] = np.loadtxt(name_seg)
            mesh.point_data['segmentation_gt'] = np.round(mesh.point_data['segmentation_gt'])
            mesh.save(join(res_dir, str(c) + '.vtk'), binary=False)
            c += 1


    shape_path = 'Datasets/Shapes/Human_pose/shapes'
    all_shapes = [el for el in listdir(shape_path) if not el.startswith('.')]

    for p in all_shapes:
        shape = pv.read(join(shape_path, p))
        new_shape = shape.copy()
        new_shape.clear_data()
        new_shape = attach_reps(new_shape)
        new_shape.cell_data['original_labels'] = shape.cell_data['original_segmentation_gt']
        new_shape.point_data['labels'] = shape.point_data['segmentation_gt']
        new_shape.save(join(shape_path, p), binary=False)

    print('creating train splits')
    all_labels = np.unique(info['dataset'])
    label_vals = np.arange(len(all_labels))
    label_dict = dict(zip(all_labels, label_vals))
    labels = [label_dict[k] for k in info['dataset']]
    info['labels'] = labels

    info = shuffle_split(info, name='split', label_name='labels')

    info.to_csv('Datasets/Shapes/Human_pose/info.csv')

    breakpoint()



if __name__ == '__main__':
    main()
