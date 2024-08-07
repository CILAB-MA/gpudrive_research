# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Dataloader for imitation learning in Nocturne."""
from collections import defaultdict
import random

import torch
from pathlib import Path
import numpy as np
from gpudrive import SimManager
from pygpudrive.env.config import SceneConfig
from pygpudrive.env.scene_selector import select_scenes
from gpudrive.madrona import ExecMode
import gpudrive
ERR_VAL = -1e4

def _get_waymo_iterator(paths, dataloader_config, scenario_config):
    # if worker has no paths, return an empty iterator
    if len(paths) == 0:
        return
    steer_actions = torch.round(
        torch.linspace(-1.0, 1.0, 13), decimals=3
    )
    accel_actions = torch.round(
        torch.linspace(-4.0, 4.0, 7), decimals=3
    )
    steer_expanded = steer_actions.view(1, 1, -1)
    accel_expanded = accel_actions.view(1, 1, -1)
    print('steer, accel', steer_actions, accel_actions)
    # load dataloader config
    tmin = dataloader_config.get('tmin', 0)
    tmax = dataloader_config.get('tmax', 90)
    dt = dataloader_config.get('dt', 0.1)
    n_stacked_states = dataloader_config.get('n_stacked_states', 5)
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    while True:
        # select a random scenario path
        # print(f'Path {len(paths)}')
        scenario_path = np.random.choice(paths)
        # print(scenario_path)
        # create simulation
        # sim = Simulation(str(scenario_path), scenario_config)
        print(f'LOAD {[str(scenario_path)]}')
        sim = SimManager(
            exec_mode=ExecMode.CPU,
            gpu_id=0,
            scenes=[str(scenario_path)],
            params=gpudrive.Parameters()
        )
        print(f'sim {sim}')
        valid_tensor = sim.shape_tensor().to_torch()
        print(f"Shape tensor has a shape of (Num Worlds, 2): {valid_tensor.shape}")
        for world_idx in range(valid_tensor.shape[0]):
            print(
                f"World {world_idx} has {valid_tensor[world_idx][0]} VALID agents and {valid_tensor[world_idx][1]} VALID road objects"
            )
        # scenario = sim.getScenario()
        controlled_state_tensor = sim.controlled_state_tensor().to_torch()
        expert_trejectory_tensor = sim.expert_trajectory_tensor().to_torch()
        print(f'controlle shape {controlled_state_tensor.shape} expert traj shape {expert_trejectory_tensor.shape}')
        # set objects to be expert-controlled

        # initialize values if stacking states
        stacked_state = defaultdict(lambda: None)
        initial_warmup = n_stacked_states - 1

        state_list = []
        action_list = []

        # iterate over timesteps and objects of interest
        n_stacked_states = 1 # todo : stacking lock
        expert_trajectory_tensor = sim.expert_trajectory_tensor().to_torch()
        num_world, num_vehicle, _ = expert_trejectory_tensor.shape
        # mask = mask.unsqueeze(-1).expand(-1, -1, expert_trajectory_tensor.shape[-1])

        print(expert_trejectory_tensor.shape)
        invActions = expert_trajectory_tensor[:, :, 6 * 91:].view(num_world, num_vehicle, 91, 3)  # todo: make those as a label
        # print(f'First veh action {invActions[:, :, 0]}')
        for time in range(tmin, tmax):
            # get state
            ego_state = sim.self_observation_tensor().to_torch() #todo : normalize
            visible_state = sim.partner_observations_tensor().to_torch() # todo : normalize
            road_map_state = sim.agent_roadmap_tensor().to_torch()
            action_tensor = sim.action_tensor().to_torch()
            print(ego_state.shape, visible_state.shape, road_map_state.shape, action_tensor.shape)
            num_world, num_vehicle, _ = ego_state.shape
            visible_state = visible_state.reshape(num_world, num_vehicle, -1)
            road_map_state = road_map_state.reshape(num_world, num_vehicle, -1)
            print(ego_state.shape, visible_state.shape, road_map_state.shape)
            state = torch.cat((ego_state, visible_state, road_map_state), axis=-1)
            mask = (sim.controlled_state_tensor().to_torch() == 1).squeeze(2)
            if n_stacked_states > 1: #todo: check
                stacked_state = torch.zeros(
                    num_world, num_vehicle, len(state) * n_stacked_states, dtype=state.dtype)
                stacked_state = torch.roll(
                    stacked_state, len(state))
                stacked_state[:len(state)] = state

            # if np.isclose(obj.position.x, ERR_VAL): # todo: find the position x in state
            #     continue

            expert_trajectory_tensor = sim.expert_trajectory_tensor().to_torch()
            mask = mask.unsqueeze(-1).expand(-1, -1, expert_trajectory_tensor.shape[-1])

            # expert_actions = torch.zeros(num_world, num_vehicle, 3)
            expert_actions_t = invActions[:, :, time]
            expert_accel = expert_actions_t[:, :, 0]
            expert_steer = expert_actions_t[:, :, 1]
            print(f'First veh steer {expert_steer} accel {expert_steer}')

            # actions[:, mask] = invActions #todo get mask
            #todo---------------------------------- start here for making expert action know indices mean

            # yield state and expert action
            if stacked_state[obj.getID()] is not None:
                if initial_warmup <= 0:  # warmup to wait for stacked state to be filled up
                    state_list.append(stacked_state[obj.getID()])
                    action_list.append(expert_action)
            else:
                state_list.append(state)
                action_list.append(expert_action)

            # step the simulation
            sim.step(dt)
            if initial_warmup > 0:
                initial_warmup -= 1

        if len(state_list) > 0:
            temp = list(zip(state_list, action_list))
            random.shuffle(temp)
            state_list, action_list = zip(*temp)
            for state_return, action_return in zip(state_list, action_list):
                yield (state_return, action_return)


class WaymoDataset(torch.utils.data.IterableDataset):
    """Waymo dataset loader."""

    def __init__(self,
                 data_path,
                 dataloader_config={},
                 scenario_config={},
                 file_limit=None):
        super(WaymoDataset).__init__()

        # save configs
        self.dataloader_config = dataloader_config
        self.scenario_config = scenario_config

        # get paths of dataset files (up to file_limit paths)
        self.file_paths = list(
            Path(data_path).glob('tfrecord*.json'))[:file_limit]
        scenario_config2 = SceneConfig(path="data", num_scenes=1)
        print(f'Scenario Config IN GPUDRIVE {scenario_config2}')
        print(f'select scene {select_scenes(scenario_config2)}')
        print(f'WaymoDataset: loading {len(self.file_paths)} files.')
        self.data_path = data_path
        # sort the paths for reproducibility if testing on a small set of files
        self.file_paths.sort()

    def __iter__(self):
        """Partition files for each worker and return an (state, expert_action) iterable."""
        # get info on current worker process
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # single-process data loading, return the whole set of files
            return _get_waymo_iterator(self.file_paths, self.dataloader_config,
                                       self.scenario_config)

        # distribute a unique set of file paths to each worker process
        worker_file_paths = np.array_split(
            self.file_paths, worker_info.num_workers)[worker_info.id]
        return _get_waymo_iterator(list(worker_file_paths),
                                   self.dataloader_config,
                                   self.scenario_config)


if __name__ == '__main__':
    dataset = WaymoDataset(data_path='dataset/tf_records',
                           file_limit=20,
                           dataloader_config={
                               'view_dist': 80,
                               'n_stacked_states': 3,
                           },
                           scenario_config={
                               'start_time': 0,
                               'allow_non_vehicles': True,
                               'spawn_invalid_objects': True,
                           })

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
    )

    for i, x in zip(range(100), data_loader):
        print(i, x[0].shape, x[1].shape)
