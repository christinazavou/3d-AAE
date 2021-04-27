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
    if dataset_name == 'contentstylecomponent':
        from datasets.contentstylecomponent import ContentStyleComponentDataset
        train_dataset = ContentStyleComponentDataset(root_dir=eval_config['data_dir'],
                                                     split='train')
        test_dataset = ContentStyleComponentDataset(root_dir=eval_config['data_dir'],
                                                     split='test')
    else:
        raise ValueError(f'Invalid dataset name. Expected `shapenet` or '
                         f'`faust`. Got: `{dataset_name}`')

    #
    # Models
    #
    arch = import_module(f"models.{eval_config['arch']}")
    CE = arch.Encoder(train_config['model']['CE']).to(device)
    SE = arch.Encoder(train_config['model']['SE']).to(device)

    #
    # Load saved state
    #
    CE.load_state_dict(torch.load(join(weights_path, f'{epoch:05}_CE.pth')))
    SE.load_state_dict(torch.load(join(weights_path, f'{epoch:05}_SE.pth')))

    CE.eval()
    SE.eval()

    train_test_sets = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    data_loader = DataLoader(train_test_sets, batch_size=eval_config['batch_size'],
                             shuffle=False, num_workers=4,
                             drop_last=False, pin_memory=True)

    with torch.no_grad():

        buildings_groups = {}
        building_components = {}
        for XC_batch, XDC_batch, XS_batch, X_batch_files in data_loader:
            XC_batch = XC_batch.to(device)

            cz_e_batch = CE(XC_batch.transpose(1, 2))
            if isinstance(cz_e_batch, tuple):
                cz_e_batch = cz_e_batch[0]

            for z_e, CX_file in zip(cz_e_batch, X_batch_files[0]):
                filename = os.path.basename(CX_file)
                building = filename.split("_style_mesh_")[0]
                component = filename.split("_style_mesh_")[1].replace("_coarse.ply", "")
                building_components.setdefault(building, [])
                if component in building_components[building]:
                    continue
                if building in buildings_groups:
                    group = buildings_groups[building] + 1
                else:
                    group = 0
                buildings_groups[building] = group
                building_components[building].append(component)
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
