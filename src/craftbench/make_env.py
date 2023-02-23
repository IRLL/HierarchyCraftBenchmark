from typing import Optional, Union
import os

import numpy as np
import gym

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

from crafting import CraftingEnv
from crafting.examples import (
    MineCraftingEnv, RandomCraftingEnv,
    TowerCraftingEnv,
    RecursiveCraftingEnv,
    LightRecursiveCraftingEnv
)



def record_wrap_env(crafting_env: CraftingEnv, video_path: str):
    crafting_env = Monitor(crafting_env)  # record stats such as returns
    env = DummyVecEnv([lambda: crafting_env])
    env = VecVideoRecorder(
        env,
        video_path,
        record_video_trigger=lambda step: (step >= 1000)
        and (int(np.log10(step + 1)) - int(np.log10(step))),
        video_length=crafting_env.max_step,
    )
    return env
