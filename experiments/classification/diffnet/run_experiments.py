import logging
import os
from os.path import join, exists
import pandas as pd
from shape_classification import run_classification_model


def main(base_fold, dataset):
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%d/%m/%Y %H:%M:%S')
    device = 'cuda'
    logging.info(f'Starting {dataset} experiment')
    data_fold = join(base_fold, dataset)
    data = join(data_fold, 'shapes')
    info = pd.read_csv(join(data_fold, 'info.csv'), index_col='idx', delimiter=';')
    res_fold = join(data_fold, 'results', 'diffusion_net')
    if not exists(res_fold):
        os.mkdir(res_fold)
    cache = join(data_fold, 'cached_data.pt')
    split_name = '10_split_'
    rr = True
    if dataset == 'Shrec_16':
        n = 30
    elif dataset == 'Atrium':
        n = 2
        res_fold = join(data_fold, 'results_classif', 'diffusion_net')
        if not exists(res_fold):
            os.mkdir(res_fold)
    else:
        n = 2

    for f in ['k1k2', 'k']:
        res = join(res_fold, f)
        run_classification_model(data_dir=data, info_df=info, res_dir=res, label_name='labels', cached_data=cache,
                                 n_fold=5, split_name_base=split_name, random_rotate=rr, label_smoothing=.2,
                                 n_class=n, input_features=f, width=64, depth=4, k_eig=128,
                                 n_epoch=100, lr=1e-3, scheduler_step=50, scheduler_gamma=.5,
                                 device_name=device)
        logging.info(f'{f} done')
    logging.info(f'{dataset} experiment is done')


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, required=True)
    # args = parser.parse_args()
    # dataset = args.dataset
    dataset='Shrec_16'
    base_fold = 'Datasets/Shapes'
    main(base_fold, dataset)
