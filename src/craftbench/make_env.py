from typing import Optional, Union
import os

import numpy as np
import gym

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

from crafting import CraftingEnv, MineCraftingEnv, RandomCraftingEnv
from crafting.examples.simple import (
    TowerCraftingEnv,
    RecursiveCraftingEnv,
    LightRecursiveCraftingEnv,
    LighterRecursiveCraftingEnv,
)

from crafting.task import RewardShaping, get_task


def choose_env(env_name: str, **kwargs):
    if env_name == "RandomCrafting-v1":
        env = RandomCraftingEnv(
            n_items=kwargs["n_items"],
            n_tools=kwargs["n_tools"],
            n_findables=kwargs["n_findables"],
            n_required_tools=[0.25, 0.4, 0.2, 0.1, 0.05],
            n_inputs_per_craft=[0.1, 0.6, 0.3],
            n_zones=kwargs["n_zones"],
            max_step=kwargs["max_episode_steps"],
            seed=kwargs["env_seed"],
        )
    elif env_name == "MineCrafting-v1":
        env = MineCraftingEnv(
            max_step=kwargs["max_episode_steps"], seed=kwargs["env_seed"]
        )
    elif env_name == "TowerCrafting-v1":
        env_kwargs = {}
        if "max_episode_steps" in kwargs:
            env_kwargs["max_episode_steps"] = kwargs["max_episode_steps"]
        env = TowerCraftingEnv(
            height=kwargs["height"],
            width=kwargs["width"] ** env_kwargs,
        )
    elif env_name == "RecursiveCrafting-v1":
        env = RecursiveCraftingEnv(
            n_items=kwargs["n_items"], reward_shaping=kwargs["reward_shaping"]
        )
    elif env_name == "LightRecursiveCrafting-v1":
        env = LightRecursiveCraftingEnv(
            n_items=kwargs["n_items"],
            n_required_previous=kwargs["n_required_previous"],
            reward_shaping=kwargs["reward_shaping"],
        )
    elif env_name == "LighterRecursiveCrafting-v1":
        env = LighterRecursiveCraftingEnv(
            n_items=kwargs["n_items"],
            n_required_previous=kwargs["n_required_previous"],
            reward_shaping=kwargs["reward_shaping"],
        )
    else:
        env = gym.make(env_name, max_step=kwargs["max_episode_steps"])

    return env


def build_task(
    env: CraftingEnv,
    reward_shaping: RewardShaping,
    task_name: Optional[str] = None,
    task_complexity: Optional[Union[int, str]] = None,
    task_seed: Optional[int] = None,
):
    reward_shaping = RewardShaping(reward_shaping)
    cache_path = os.path.join("cache", f"{env.name}_{env.original_seed}")
    if isinstance(task_complexity, str):
        if task_complexity.isdigit():
            task_complexity = int(task_complexity)
        else:
            task_complexity = None

    if task_complexity is not None:
        task_name = None

    return get_task(
        world=env.world,
        task_name=task_name,
        task_complexity=task_complexity,
        cache_path=cache_path,
        reward_shaping=reward_shaping,
        seed=task_seed,
    )


def make_env(config: dict):
    env = choose_env(**config)
    reward_shaping = config["reward_shaping"]
    task_name = config.get("task_name")
    task_complexity = config.get("task_complexity")
    if task_name is not None or task_complexity is not None:
        task = build_task(
            env,
            reward_shaping,
            task_name,
            task_complexity,
        )
        env.add_task(task, can_end=True)
    task = env.tasks[-1]
    print(f"{task=} | {reward_shaping=}")
    if "task" not in config:
        config["task"] = task.name
    return env


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
