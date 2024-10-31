import os

import pyvista as pv
import numpy as np
import pandas as pd
from os import listdir
from os.path import join
from cleaner_tools import attach_reps, fix_off, shuffle_split


def main():
    breakpoint()
    init_path = 'Datasets/Shapes/Shrec_16'
    fold_path = join(init_path, 'original_coarse')
    fold_fine_path = join(init_path, 'original')
    info = pd.DataFrame({'name': [], 'init_split': [], 'class': []}, index=[])

    final_shape_path = join(init_path, 'shapes_coarse')
    final_fine_shape_path = join(init_path, 'shapes')
    if not os.path.exists(final_shape_path):
        os.mkdir(final_shape_path)
    if not os.path.exists(final_fine_shape_path):
        os.mkdir(final_fine_shape_path)

    all_classes = [el for el in listdir(fold_path) if not el.startswith('.')]
    idx = 0
    pd.re
    n_class = len(all_classes)
    n = 0
    for c in all_classes:
        n += 1
        test_fold = [e for e in listdir(join(fold_path, c, 'test')) if not e.startswith('.')]
        train_fold = [e for e in listdir(join(fold_path, c, 'train')) if not e.startswith('.')]
        print(f'Doing class {c}, number {n}/{n_class}.')
        print(f'Train size is {len(train_fold)}')
        split_val = 1  # train
        for case in train_fold:
            case_name = case.split('.')[0]
            info.loc[idx] = [case_name, split_val, c]
            shape = attach_reps(join(fold_path, c, 'train', case))
            shape.save(join(final_shape_path, str(idx) + '.vtk'), binary=False)
            fix_off(join(fold_fine_path, case_name + '.off'))
            shape_fine = attach_reps(join(fold_fine_path, case_name + '.off'))
            shape_fine.save(join(final_fine_shape_path, str(idx) + '.vtk'), binary=False)
            idx += 1

        print(f'test size is {len(test_fold)}')
        split_val = 0  # test
        for case in test_fold:
            case_name = case.split('.')[0]
            info.loc[idx] = [case_name, split_val, c]
            shape = attach_reps(join(fold_path, c, 'test', case))
            shape.save(join(final_shape_path, str(idx) + '.vtk'), binary=False)
            fix_off(join(fold_fine_path, case_name+'.off'))
            shape_fine = attach_reps(join(fold_fine_path, case_name + '.off'))
            shape_fine.save(join(final_fine_shape_path, str(idx) + '.vtk'), binary=False)
            idx += 1
    info.index.name = 'idx'

    print('creating train splits')
    all_labels = np.unique(info['class'])
    label_vals = np.arange(len(all_labels))
    label_dict = dict(zip(all_labels, label_vals))
    labels = [label_dict[k] for k in info['class']]
    info['labels'] = labels

    info = shuffle_split(info, name='full_split', label_name='labels')

    info.to_csv(join(init_path, 'info.csv'))

    print('done')
    breakpoint()


if __name__ == "__main__":
    main()
