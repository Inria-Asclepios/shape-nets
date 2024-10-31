from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from shapecentral import Surface
import numpy as np


def shuffle_split(frame, sub_frame=None, n_split=5, test_size=.2, name='train_split', label_name=None):
    new_frame = frame.copy()
    if sub_frame is not None:
        new_sub_frame = sub_frame.copy()
        idx = new_sub_frame.index
        if label_name is None:
            stratified = None
        else:
            stratified = new_sub_frame[label_name]
    else:
        idx = new_frame.index
        if label_name is None:
            stratified = None
        else:
            stratified = new_frame[label_name]

    if stratified is not None:
        kf = StratifiedShuffleSplit(n_splits=n_split, test_size=test_size)
        splits = [np.isin(idx, np.array(idx)[train]).astype(int) for train, _ in kf.split(idx, stratified)]
    else:
        kf = ShuffleSplit(n_splits=n_split, test_size=test_size)
        splits = [np.isin(idx, np.array(idx)[train]).astype(int) for train, _ in kf.split(idx)]

    for i in range(n_split):
        if sub_frame is not None:
            new_splits = np.zeros(len(new_frame.index))
            new_splits[idx] = splits[i]
        else:
            new_splits = splits[i]
        new_frame[name + '_' + str(i)] = new_splits

    return new_frame


def sub_sample(frame, class_names, n_per_class=20, ids=None, name='lite'):
    new_frame = frame.copy()
    if ids is not None:
        new_frame.set_index(ids)
        print(f'The ids name {ids} were set as dataframe index')

    all_selected_ids = []
    all_classes = np.unique(frame[class_names])
    for c in all_classes:
        idx = frame[frame[class_names] == c].index
        rng = np.random.default_rng()
        selected_ids = rng.choice(idx, n_per_class, replace=False)
        all_selected_ids.append(selected_ids)
    subsamples = np.zeros(len(new_frame.index))
    subsamples[all_selected_ids] = 1
    new_frame[name] = subsamples
    return new_frame


def attach_reps(shape, principal_from_kh=False):
    surf = Surface()
    surf.load_shape(shape)
    new_shape = surf.get_all_reps(principal_from_kh=principal_from_kh)
    return new_shape

def random_transform(shape):
    rng = np.random.default_rng()
    rotation_vector = rng.integers(1, 100, size=3)
    rotation_degree = rng.integers(1, 180, size=1)[0]
    translation_vector = rng.integers(1, size=3)

    trans_shape = shape.translate(translation_vector, inplace=False)
    rot_trans_shape = trans_shape.rotate_vector(rotation_vector, rotation_degree, inplace=False)
    return rot_trans_shape

def fix_off(filename):
    f = open(filename)
    first_line, remainder = f.readline(), f.read()
    if first_line != 'OFF\n':
        new_line = 'OFF\n ' + first_line.split('OFF')[1]
        t = open(filename, "w")
        t.write(new_line)
        t.write(remainder)
        t.close()
        print(f'cleaned {filename}')


def edge_to_point_data(shape, edge_data, data_name='labels'):
    new_shape = shape.copy()
    edges = new_shape.extract_all_edges()
    edges.clear_data()
    edges.cell_data['labels'] = edge_data
    edges = edges.cell_data_to_point_data()
    new_shape.point_data['labels'] = edges.point_data['labels']
    return new_shape
