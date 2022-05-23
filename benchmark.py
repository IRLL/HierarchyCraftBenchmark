import os
import time

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

from sb3_contrib.ppo_mask.ppo_mask import MaskablePPO

from option_graph.metrics.complexity import learning_complexity
from option_graph.metrics.complexity.histograms import nodes_histograms
from option_graph.option import Option

import wandb

from crafting import CraftingEnv, MineCraftingEnv, RandomCraftingEnv
from crafting.task import (
    RewardShaping,
    TaskObtainItem,
    adaptative_max_episode_step,
    get_task,
)

from callbacks import WandbCallback
from plots import save_requirement_graph, save_option_graph

if __name__ == "__main__":

    env_name = "RandomCrafting-v1"
    project = "crafting-benchmark"

    config = {
        "agent": "MaskablePPO",
        "agent_seed": 0,
        "policy_type": "MlpPolicy",
        "pi_n_layers": 3,
        "pi_units_per_layer": 64,
        "vf_n_layers": 3,
        "vf_units_per_layer": 64,
        "total_timesteps": 1e6,
        "max_n_consecutive_successes": 100,
        "env_name": env_name,
        "env_seed": 1,
        "task_seed": None,
        "task_complexity": 243,
        "reward_shaping": RewardShaping.ALL_USEFUL.value,
        "max_episode_steps": None,
        "time_factor": 2.0,
        "n_items": 50,
        "n_tools": 0,
        "n_findables": 5,
        "n_zones": 1,
    }

    run = wandb.init(project=project, config=config, monitor_gym=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dirname = f"{timestamp}-{run.id}"
    config = wandb.config
    config["max_episode_steps"] = (
        config["max_episode_steps"]
        if isinstance(config["max_episode_steps"], int)
        else None
    )

    def make_env():
        if config["env_name"] == "RandomCrafting-v1":
            env = RandomCraftingEnv(
                n_items=config["n_items"],
                n_tools=config["n_tools"],
                n_findables=config["n_findables"],
                n_required_tools=[0.25, 0.4, 0.2, 0.1, 0.05],
                n_inputs_per_craft=[0.1, 0.6, 0.3],
                n_zones=config["n_zones"],
                max_step=config["max_episode_steps"],
                seed=config["env_seed"],
            )
        elif config["env_name"] == "MineCrafting-v1":
            env = MineCraftingEnv(
                max_step=config["max_episode_steps"],
                seed=config["env_seed"],
            )
        else:
            raise ValueError

        reward_shaping = RewardShaping(config["reward_shaping"])

        if config["task_complexity"] is not None:
            config["task_seed"] = None
            task = get_task(
                world=env.world,
                task_complexity=config["task_complexity"],
                cache_path=os.path.join(
                    "cache", f'{config["env_name"]}_{config["env_seed"]}'
                ),
                reward_shaping=reward_shaping,
            )
        else:
            task = get_task(
                world=env.world,
                task_name=config["task"],
                reward_shaping=reward_shaping,
                seed=config["task_seed"],
            )
        print(f"{task=} | {reward_shaping=}")
        config["task"] = task.name

        if config["max_episode_steps"] is None:
            adaptative_max_step = adaptative_max_episode_step(
                env.world, task, time_factor=config["time_factor"]
            )
            env.max_step = config["max_episode_steps"] = adaptative_max_step

        env.add_task(task, can_end=True)
        env = Monitor(env)  # record stats such as returns
        return env

    env = DummyVecEnv([make_env])
    crafting_env: CraftingEnv = env.envs[0]
    env = VecVideoRecorder(
        env,
        f"videos/{run.id}",
        record_video_trigger=lambda step: step % 10000 == 0,
        video_length=config["max_episode_steps"],
    )

    task: TaskObtainItem = crafting_env.tasks[0]

    # Get & save requirements graph
    requirement_graph_path = save_requirement_graph(
        run_dirname, crafting_env.world, title=str(crafting_env.world), figsize=(32, 18)
    )

    # Get & save solving option
    all_options = crafting_env.world.get_all_options()
    all_options_list = list(all_options.values())
    solving_option: Option = all_options[f"Get {task.goal_item}"]
    solving_option_graph_path = save_option_graph(solving_option, run_dirname)

    # Compute complexities
    used_nodes_all = nodes_histograms(all_options_list)
    lcomp, comp_saved = learning_complexity(solving_option, used_nodes_all)
    print(f"OPTION: {str(solving_option)}: {lcomp} ({comp_saved})")

    pi_arch = [config["pi_units_per_layer"] for _ in range(config["pi_n_layers"])]
    vf_arch = [config["vf_units_per_layer"] for _ in range(config["vf_n_layers"])]
    net_arch = [dict(pi=pi_arch, vf=vf_arch)]
    agent = MaskablePPO(
        config["policy_type"],
        env,
        policy_kwargs={"net_arch": net_arch},
        seed=config["agent_seed"],
        verbose=1,
    )

    wandb.log(
        {
            "task": str(task),
            "solving_option": str(solving_option),
            "learning_complexity": lcomp,
            "total_complexity": lcomp + comp_saved,
            "saved_complexity": comp_saved,
            "requirement_graph": wandb.Image(requirement_graph_path),
            "solving_option_graph": wandb.Image(solving_option_graph_path),
        }
    )

    # pylint: disable=unexpected-keyword-arg
    agent.learn(
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(
            verbose=2,
            max_n_consecutive_successes=config["max_n_consecutive_successes"],
        ),
    )

    run.finish()
