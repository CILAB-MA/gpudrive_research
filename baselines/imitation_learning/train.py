# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Imitation learning training script (behavioral cloning)."""
from datetime import datetime
from pathlib import Path
import pickle
import random
import json
import os, sys
sys.path.append(os.getcwd())
# import hydra
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from baselines.imitation_learning.model import ImitationAgent
from baselines.imitation_learning.waymo_data_loader import WaymoDataset


def set_seed_everywhere(seed):
    """Ensure determinism."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# @hydra.main(config_path="../../cfgs/imitation", config_name="config")
def main():
    """Train an IL model."""
    base_path = '/data/formatted_json_v2_no_tl_'
    train_path = base_path + 'train'
    validation_path = base_path + 'validation'
    test_path = base_path + 'testing'

    args = dict(
        seed=0,
        use_wandb=False,
        samples_per_epcoh=50000,
        batch_size=512,
        epochs=700,
        num_files=1000,
        path=train_path
    )
    set_seed_everywhere(args['seed'])
    # create dataset and dataloader
    expert_bounds = [[-6, 6], [-0.7, 0.7]]
    actions_bounds = expert_bounds
    actions_discretizations = [15, 43]
    mean_scalings = [3, 0.7]
    std_devs = [0.1, 0.02]

    dataloader_cfg = {
        'tmin': 0,
        'tmax': 90,
        'view_dist': 80,
        'view_angle': 2.1,
        'dt': 0.1,
        'expert_action_bounds': expert_bounds,
        'expert_position': False,
        'state_normalization': 100,
        'n_stacked_states': 5,
    }
    scenario_cfg = {
        'start_time': 0,
        'allow_non_vehicles': True,
        'spawn_invalid_objects': True,
        'max_visible_road_points': 500,
        'sample_every_n': 1,
        'road_edge_first': False,
    }
    dataset = WaymoDataset(
        data_path=args['path'],
        file_limit=args['num_files'],
        dataloader_config=dataloader_cfg,
        scenario_config=scenario_cfg,
    )
    data_loader = iter(
        DataLoader(
            dataset,
            batch_size=args['batch_size'],
            num_workers=10,
            pin_memory=True,
        ))

    # create model
    sample_state, _ = next(data_loader)
    n_states = sample_state.shape[-1]

    model_cfg = {
        'n_inputs': n_states,
        'hidden_layers': [1024, 256, 128],
        'discrete': True,
        'mean_scalings': mean_scalings,
        'std_devs': std_devs,
        'actions_discretizations': actions_discretizations,
        'actions_bounds': actions_bounds,
        'device': 'cuda:0'
    }

    model = ImitationAgent(model_cfg).to('cuda:0')
    model.train()
    print(model)

    # create optimizer
    optimizer = Adam(model.parameters(), lr=3e-4)

    # create exp dir
    time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    exp_dir = Path.cwd() / Path('train_logs') / time_str
    exp_dir.mkdir(parents=True, exist_ok=True)

    # save configs
    configs_path = exp_dir / 'configs.json'
    configs = {
        'scenario_cfg': scenario_cfg,
        'dataloader_cfg': dataloader_cfg,
        'model_cfg': model_cfg,
    }
    with open(configs_path, 'w') as fp:
        json.dump(configs, fp, sort_keys=True, indent=4)
    print('Wrote configs at', configs_path)

    # tensorboard writer
    writer = SummaryWriter(log_dir=str(exp_dir))
    # todo: wandb logging

    # training loop
    print('Exp dir created at', exp_dir)
    print(f'`tensorboard --logdir={exp_dir}`\n')
    for epoch in range(args['epochs']):
        print(f'\nepoch {epoch+1}/{args["epochs"]}')
        n_samples = epoch * args['batch_size'] * (args['samples_per_epoch'] //
                                               args['batch_size'])

        for i in tqdm(range(args['samples_per_epoch'] // args['batch_size']),
                      unit='batch'):
            # get states and expert actions
            states, expert_actions = next(data_loader)
            states = states.to('cuda:0')
            expert_actions = expert_actions.to('cuda:0')

            # compute loss
            if args['discrete']:
                log_prob, expert_idxs = model.log_prob(states,
                                                       expert_actions,
                                                       return_indexes=True)
            else:
                dist = model.dist(states)
                log_prob = dist.log_prob(expert_actions.float())
            loss = -log_prob.mean()

            metrics_dict = {}

            # optim step
            optimizer.zero_grad()
            loss.backward()

            # grad clipping
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item()**2
            total_norm = total_norm**0.5
            metrics_dict['training/grad_norm'] = total_norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item()**2
            total_norm = total_norm**0.5
            metrics_dict['training/post_clip_grad_norm'] = total_norm
            optimizer.step()

            # tensorboard logging
            metrics_dict['training/loss'] = loss.item()

            metrics_dict['training/accel_logprob'] = log_prob[0]
            metrics_dict['training/steer_logprob'] = log_prob[1]

            if not model_cfg['discrete']:
                diff_actions = torch.mean(torch.abs(dist.mean -
                                                    expert_actions),
                                          axis=0)
                metrics_dict['training/accel_diff'] = diff_actions[0]
                metrics_dict['training/steer_diff'] = diff_actions[1]
                metrics_dict['training/l2_dist'] = torch.norm(
                    dist.mean - expert_actions.float())

            if model_cfg['discrete']:
                with torch.no_grad():
                    model_actions, model_idxs = model(states,
                                                      deterministic=True,
                                                      return_indexes=True)
                accuracy = [
                    (model_idx == expert_idx).float().mean(axis=0)
                    for model_idx, expert_idx in zip(model_idxs, expert_idxs.T)
                ]
                metrics_dict['training/accel_acc'] = accuracy[0]
                metrics_dict['training/steer_acc'] = accuracy[1]

            for key, val in metrics_dict.items():
                writer.add_scalar(key, val, n_samples)
            if args['use_wandb']:
                wandb.log(metrics_dict, step=n_samples)
        # save model checkpoint
        if (epoch + 1) % 10 == 0 or epoch == args['epochs'] - 1:
            model_path = exp_dir / f'model_{epoch+1}.pth'
            torch.save(model, str(model_path))
            pickle.dump(filter, open(exp_dir / f"filter_{epoch+1}.pth", "wb"))
            print(f'\nSaved model at {model_path}')
        if args['discrete']:
            print('accel')
            print('model: ', model_idxs[0][0:10])
            print('expert: ', expert_idxs[0:10, 0])
            print('steer')
            print('model: ', model_idxs[1][0:10])
            print('expert: ', expert_idxs[0:10, 1])

    print('Done, exp dir is', exp_dir)

    writer.flush()
    writer.close()


if __name__ == '__main__':
    main()
