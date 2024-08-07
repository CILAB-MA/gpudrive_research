import torch
import numpy as np
import gpudrive

max_speed = 100
max_veh_len = 30
max_veh_width = 10
min_rel_goal_coord = -1000
max_rel_goal_coord = 1000
min_rel_agent_pos = -1000
max_rel_agent_pos = 1000
max_orientation_rad = 2 * np.pi
min_rm_coord = -1000
max_rm_coord = 1000
max_road_line_segmment_len = 100
max_road_scale = 100
ROAD_OBJECT_TYPES = 4

ENTITY_TYPE_TO_INT = {
            gpudrive.EntityType._None: 0,
            gpudrive.EntityType.RoadEdge: 1,
            gpudrive.EntityType.RoadLine: 2,
            gpudrive.EntityType.RoadLane: 3,
            gpudrive.EntityType.CrossWalk: 4,
            gpudrive.EntityType.SpeedBump: 5,
            gpudrive.EntityType.StopSign: 6,
            gpudrive.EntityType.Vehicle: 7,
            gpudrive.EntityType.Pedestrian: 8,
            gpudrive.EntityType.Cyclist: 9,
            gpudrive.EntityType.Padding: 10,
        }
MIN_OBJ_ENTITY_ENUM = min(list(ENTITY_TYPE_TO_INT.values()))
MAX_OBJ_ENTITY_ENUM = max(list(ENTITY_TYPE_TO_INT.values()))

def normalize_tensor(x, min_val, max_val):
    return 2 * ((x - min_val) / (max_val - min_val)) - 1

def normalize_ego_state(state):
    """Normalize ego state features."""

    # Speed, vehicle length, vehicle width
    state[:, :, 0] /= max_speed
    state[:, :, 1] /= max_veh_len
    state[:, :, 2] /= max_veh_width

    # Relative goal coordinates
    state[:, :, 3] = normalize_tensor(
        state[:, :, 3],
        min_rel_goal_coord,
        max_rel_goal_coord,
    )
    state[:, :, 4] = normalize_tensor(
        state[:, :, 4],
        min_rel_goal_coord,
        max_rel_goal_coord,
    )
    return state

def normalize_and_flatten_partner_obs(obs):
    """Normalize partner state features.
    Args:
        obs: torch.Tensor of shape (num_worlds, kMaxAgentCount, kMaxAgentCount - 1, num_features)
    """
    obs = torch.nan_to_num(obs, nan=0)

    # Speed
    obs[:, :, :, 0] /= max_speed

    # Relative position
    obs[:, :, :, 1] = normalize_tensor(
        obs[:, :, :, 1],
        min_rel_agent_pos,
        max_rel_agent_pos,
    )
    obs[:, :, :, 2] = normalize_tensor(
        obs[:, :, :, 2],
        min_rel_agent_pos,
        max_rel_agent_pos,
    )

    # Orientation (heading)
    obs[:, :, :, 3] /= max_orientation_rad

    # Vehicle length and width
    obs[:, :, :, 4] /= max_veh_len
    obs[:, :, :, 5] /= max_veh_width

    # One-hot encode the type of the other visible objects
    one_hot_encoded_object_types = one_hot_encode_object_type(obs[:, :, :, 6])

    # Concat the one-hot encoding with the rest of the features
    obs = torch.concat((obs[:, :, :, :6], one_hot_encoded_object_types), dim=-1)

    return obs.flatten(start_dim=2)

def one_hot_encode_object_type(object_type_tensor):
    """One-hot encode the object type."""

    VEHICLE = ENTITY_TYPE_TO_INT[gpudrive.EntityType.Vehicle]
    PEDESTRIAN = ENTITY_TYPE_TO_INT[gpudrive.EntityType.Pedestrian]
    CYCLIST = ENTITY_TYPE_TO_INT[gpudrive.EntityType.Cyclist]
    PADDING = ENTITY_TYPE_TO_INT[gpudrive.EntityType._None]

    # Set garbage object elements to zero
    object_types = torch.where(
        (object_type_tensor < MIN_OBJ_ENTITY_ENUM) | (object_type_tensor > MAX_OBJ_ENTITY_ENUM),
        0.0,
        object_type_tensor,
    ).int()

    one_hot_object_type = torch.nn.functional.one_hot(
        torch.where(
            condition=(object_types == VEHICLE) | (object_types == PEDESTRIAN) | (
                        object_types == CYCLIST) | object_types == PADDING,
            input=object_types,
            other=0,
        ).long(),
        num_classes=ROAD_OBJECT_TYPES,
    )
    return one_hot_object_type

def normalize_and_flatten_map_obs(obs):
    """Normalize map observation features."""

    # Road point coordinates
    obs[:, :, :, 0] = normalize_tensor(
        obs[:, :, :, 0],
        min_rm_coord,
        max_rm_coord,
    )

    obs[:, :, :, 1] = normalize_tensor(
        obs[:, :, :, 1],
        min_rm_coord,
        max_rm_coord,
    )

    # Road line segment length
    obs[:, :, :, 2] /= max_road_line_segmment_len

    # Road scale (width and height)
    obs[:, :, :, 3] /= max_road_scale
    # obs[:, :, :, 4] seems already scaled

    # Road point orientation
    obs[:, :, :, 5] /= max_orientation_rad

    # Road types: one-hot encode them
    one_hot_road_types = one_hot_encode_roadpoints(obs[:, :, :, 6])

    # Concatenate the one-hot encoding with the rest of the features
    obs = torch.cat((obs[:, :, :, :6], one_hot_road_types), dim=-1)

    return obs.flatten(start_dim=2)
