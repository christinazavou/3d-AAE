import argparse
import json
import logging
import os.path
import random
from importlib import import_module
from os.path import join

import numpy as np
import torch
from torch.distributions import Beta, Bernoulli
from torch.utils.data import DataLoader

from datasets.shapenet import ShapeNetDataset
from utils.util import find_latest_epoch, cuda_setup, setup_logging


def main(eval_config):
    # Load hyperparameters as they were during training
    train_results_path = join(eval_config['results_root'], eval_config['arch'],
                              eval_config['experiment_name'])
    with open(join(train_results_path, 'config.json')) as f:
        train_config = json.load(f)

    random.seed(train_config['seed'])
    torch.manual_seed(train_config['seed'])
    torch.cuda.manual_seed_all(train_config['seed'])

    log = logging.getLogger(__name__)

    weights_path = join(train_results_path, 'weights')
    if "epoch" not in eval_config or eval_config['epoch'] == 0:
        epoch = find_latest_epoch(weights_path)
    else:
        epoch = eval_config['epoch']
    log.debug(f'Starting from epoch: {epoch}')

    encodings_path = join(train_results_path, 'encodings', f'{epoch:05}_z_e')
    os.makedirs(encodings_path, exist_ok=True)

    device = cuda_setup(eval_config['cuda'])
    log.debug(f'Device variable: {device}')
    if device.type == 'cuda':
        log.debug(f'Current CUDA device: {torch.cuda.current_device()}')

    #
    # Dataset
    #
    dataset_name = train_config['dataset'].lower()
    if dataset_name == 'shapenet':
        dataset = ShapeNetDataset(root_dir=eval_config['data_dir'],
                                  classes=train_config['classes'], split='test')
    elif dataset_name == 'annfasscomponent':
        from datasets.annfasscomponent import AnnfassComponentDataset
        dataset = AnnfassComponentDataset(root_dir=eval_config['data_dir'], split='train',
                                          classes=train_config['classes'], n_points=train_config['n_points'])
    else:
        raise ValueError(f'Invalid dataset name. Expected `shapenet` or '
                         f'`faust`. Got: `{dataset_name}`')
    classes_selected = ('all' if not train_config['classes']
                        else ','.join(train_config['classes']))
    log.debug(f'Selected {classes_selected} classes. Loaded {len(dataset)} '
              f'samples.')

    #
    # Models
    #
    arch = import_module(f"models.{eval_config['arch']}")
    E = arch.Encoder(train_config).to(device)

    #
    # Load saved state
    #
    E.load_state_dict(torch.load(join(weights_path, f'{epoch:05}_E.pth')))

    E.eval()

    num_samples = len(dataset)
    data_loader = DataLoader(dataset, batch_size=eval_config['batch_size'],
                             shuffle=False, num_workers=4,
                             drop_last=False, pin_memory=True)

    with torch.no_grad():

        buildings_groups = {}
        for X_batch, X_batch_indices in data_loader:
            X_batch = X_batch.to(device)

            z_e_batch = E(X_batch.transpose(1, 2))
            if isinstance(z_e_batch, tuple):
                z_e_batch = z_e_batch[0]

            for z_e, X_idx in zip(z_e_batch, X_batch_indices):
                X_file = dataset.files[X_idx]
                filename = os.path.basename(X_file)
                building = filename.split("_style_mesh_")[0]
                if building in buildings_groups:
                    group = buildings_groups[building] + 1
                else:
                    group = 0
                buildings_groups[building] = group
                component = filename.split("_style_mesh_")[1].replace("_detail.ply", "")
                os.makedirs(os.path.join(encodings_path, building), exist_ok=True)
                np.save(join(encodings_path, building, f"group{group}_{component}.npy"), z_e.cpu().numpy())


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='File path for evaluation config')
    args = parser.parse_args()

    evaluation_config = None
    if args.config is not None and args.config.endswith('.json'):
        with open(args.config) as f:
            evaluation_config = json.load(f)
    assert evaluation_config is not None

    main(evaluation_config)
