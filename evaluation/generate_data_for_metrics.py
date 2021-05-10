import argparse
import json
import logging
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

    setup_logging(join(train_results_path, 'results'))
    log = logging.getLogger(__name__)

    weights_path = join(train_results_path, 'weights')
    if "epoch" not in eval_config or eval_config['epoch'] == 0:
        epoch = find_latest_epoch(weights_path)
    else:
        epoch = eval_config['epoch']
    log.debug(f'Starting from epoch: {epoch}')

    device = cuda_setup(eval_config['cuda'])
    log.debug(f'Device variable: {device}')
    if device.type == 'cuda':
        log.debug(f'Current CUDA device: {torch.cuda.current_device()}')

    #
    # Dataset
    #
    from datasets import load_dataset_class
    dset_class = load_dataset_class(train_config['dataset'])
    dataset = dset_class(eval_config['data_dir'], **train_config["test_dataset"])
    # val_dataset = dset_class(root_dir=config['data_dir'], classes=config['classes'], split='valid')

    log.debug("Selected {} classes. Loaded {} samples.".format(
        'all' if not train_config["test_dataset"]['classes'] else ','.join(train_config["test_dataset"]['classes']),
        len(dataset)))

    points_dataloader = DataLoader(dataset, batch_size=train_config['batch_size'],
                                   shuffle=train_config["test_dataset"]['shuffle'],
                                   num_workers=train_config['num_workers'],
                                   drop_last=True, pin_memory=True)
    classes_selected = ('all' if not train_config['classes']
                        else ','.join(train_config['classes']))
    log.debug(f'Selected {classes_selected} classes. Loaded {len(dataset)} '
              f'samples.')

    if 'distribution' in train_config:
        distribution = train_config['distribution']
    elif 'distribution' in eval_config:
        distribution = eval_config['distribution']
    else:
        log.warning('No distribution type specified. Assumed normal = N(0, 0.2)')
        distribution = 'normal'

    #
    # Models
    #
    arch = import_module(f"models.{eval_config['arch']}")
    E = arch.Encoder(train_config).to(device)
    G = arch.Generator(train_config).to(device)

    #
    # Load saved state
    #
    E.load_state_dict(torch.load(join(weights_path, f'{epoch:05}_E.pth')))
    G.load_state_dict(torch.load(join(weights_path, f'{epoch:05}_G.pth')))

    E.eval()
    G.eval()

    num_samples = len(dataset)
    data_loader = DataLoader(dataset, batch_size=num_samples,
                             shuffle=False, num_workers=4,
                             drop_last=False, pin_memory=True)

    # We take 3 times as many samples as there are in test data in order to
    # perform JSD calculation in the same manner as in the reference publication
    noise = torch.FloatTensor(3 * num_samples, train_config['z_size'], 1)
    noise = noise.to(device)

    X, _ = next(iter(data_loader))
    X = X.to(device)

    np.save(join(train_results_path, 'results', f'{epoch:05}_X'), X.cpu().numpy())

    for i in range(3):
        if distribution == 'normal':
            noise.normal_(0, 0.2)
        elif distribution == 'beta':
            noise_np = np.random.beta(train_config['z_beta_a'],
                                      train_config['z_beta_b'],
                                      noise.shape)
            noise = torch.tensor(noise_np).float().round().to(device)
        elif distribution == 'bernoulli':
            p = torch.tensor(train_config['p']).to(device)
            sampler = Bernoulli(probs=p)
            noise = sampler.sample(noise.shape)

        with torch.no_grad():
            X_g = G(noise)
        if X_g.shape[-2:] == (3, 2048):
            X_g.transpose_(1, 2)

        np.save(join(train_results_path, 'results', f'{epoch:05}_Xg_{i}'), X_g.cpu().numpy())

    with torch.no_grad():
        z_e = E(X.transpose(1, 2))
        if isinstance(z_e, tuple):
            z_e = z_e[0]
        X_rec = G(z_e)
    if X_rec.shape[-2:] == (3, 2048):
        X_rec.transpose_(1, 2)

    np.save(join(train_results_path, 'results', f'{epoch:05}_Xrec'), X_rec.cpu().numpy())


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
