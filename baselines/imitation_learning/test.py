import gpudrive

from pygpudrive.env.config import SceneConfig
from pygpudrive.env.scene_selector import select_scenes

scene_config = SceneConfig(path="data", num_scenes=1)

sim = gpudrive.SimManager(
    exec_mode=gpudrive.madrona.ExecMode.CPU, # Specify the execution mode
    gpu_id=0,
    scenes=select_scenes(scene_config),
    params=gpudrive.Parameters(),  # Environment parameters
)