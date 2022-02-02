import time

from crafting import CraftingEnv, MineCraftingEnv, RandomCraftingEnv
from crafting.task import TaskObtainItem

from option_graph.metrics.complexity import learning_complexity
from option_graph.metrics.complexity.histograms import nodes_histograms
from option_graph.option import Option

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from sb3_contrib.ppo_mask.ppo_mask import MaskablePPO

import wandb
from callbacks import WandbCallback

from plots import save_requirement_graph, save_option_graph

if __name__ == "__main__":

    MINECRAFTING = False

    if MINECRAFTING:
        env_name = "MineCrafting-v1"
        task_name = "obtain_book"
    else:
        env_name = "RandomCrafting-v1"
        task_name = "random_item"

    config = {
        "agent": "MaskablePPO",
        "policy_type": "MlpPolicy",
        "total_timesteps": 2e5,
        "max_n_consecutive_successes": 100,
        "env_name": env_name,
        "max_episode_steps": 50,
        "n_items": 50,
        "n_tools": 0,
        "n_foundables": 5,
        "n_zones": 1,
        "task": task_name,
    }

    def make_env():
        if config["env_name"] == "RandomCrafting-v1":
            env = RandomCraftingEnv(
                n_items=config["n_items"],
                n_tools=config["n_tools"],
                n_foundables=config["n_foundables"],
                n_required_tools=[0.25, 0.4, 0.2, 0.1, 0.05],
                n_inputs_per_craft=[0.1, 0.6, 0.3],
                n_zones=config["n_zones"],
                tasks=[config["task"]],
                tasks_can_end=[True],
                max_step=config["max_episode_steps"],
            )
        elif config["env_name"] == "MineCrafting-v1":
            env = MineCraftingEnv(
                tasks=[config["task"]],
                tasks_can_end=[True],
                max_step=config["max_episode_steps"],
            )
        else:
            raise ValueError
        env = Monitor(env)  # record stats such as returns
        return env

    if config["env_name"] == "RandomCrafting-v1":
        project = "crafting-benchmark"
    elif config["env_name"] == "MineCrafting-v1":
        project = "minecrafting-benchmark"

    run = wandb.init(project=project, config=config, monitor_gym=True)
    config = wandb.config

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
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dirname = f"{timestamp}-{run.id}"
    requirement_graph_path = save_requirement_graph(
        run_dirname, crafting_env.world, title=str(task), figsize=(32, 18)
    )

    # Get & save solving option
    all_options = crafting_env.world.get_all_options()
    all_options_list = list(all_options.values())
    solving_option: Option = all_options[f"Get {task.goal_item}"]
    solving_option_graph_path = save_option_graph(solving_option, run_dirname)

    # Compute complexities
    used_nodes_all = nodes_histograms(all_options_list)
    lcomp, comp_saved = learning_complexity(solving_option, used_nodes_all)
    print(f"OPTION: {str(solving_option)}: {lcomp:.2f} ({comp_saved:.2f})")

    wandb.log(
        {
            "task": str(task),
            "solving_option": str(solving_option),
            "learning_complexity": lcomp,
            "total_complexity": lcomp + comp_saved,
            "saved_complexity": comp_saved,
            "requirement_graph": wandb.Image(requirement_graph_path),
            "solving_option": wandb.Image(solving_option_graph_path),
        }
    )

    agent = MaskablePPO(config["policy_type"], env, verbose=1)
    agent.learn(
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(
            verbose=2, max_n_consecutive_successes=config["max_n_consecutive_successes"]
        ),
    )
    run.finish()
