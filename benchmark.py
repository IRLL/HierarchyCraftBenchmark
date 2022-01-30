# import wandb
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np

from crafting import CraftingEnv, RandomCraftingEnv, MineCraftingEnv
from crafting.task import TaskObtainItem
from option_graph.graph import compute_levels

from option_graph.metrics.complexity import learning_complexity
from option_graph.metrics.complexity.histograms import nodes_histograms
from option_graph.option import Option

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from sb3_contrib.ppo_mask.ppo_mask import MaskablePPO

import wandb
from wandb.integration.sb3 import WandbCallback


class WandbCallback(WandbCallback):
    def __init__(
        self,
        verbose: int = 0,
        model_save_path: str = None,
        model_save_freq: int = 0,
        gradient_save_freq: int = 0,
        max_n_consecutive_successes: Optional[int] = None,
    ):
        super().__init__(verbose, model_save_path, model_save_freq, gradient_save_freq)
        self.n_successes = 0
        self.n_consecutive_successes = 0
        self.max_n_consecutive_successes = max_n_consecutive_successes
        self._tasks_done = False

    def _on_step(self):
        # Checking for both 'done' and 'dones' keywords because:
        # Some models use keyword 'done' (e.g.,: SAC, TD3, DQN, DDPG)
        # While some models use keyword 'dones' (e.g.,: A2C, PPO)
        done = np.array(
            self.locals.get("done")
            if self.locals.get("done") is not None
            else self.locals.get("dones")
        )
        env_done = np.any(done)
        tasks_done = self.locals.get("infos")[0].get("tasks_done")
        if tasks_done:
            self._tasks_done = True
        if env_done:
            if self._tasks_done:
                self.n_successes += 1
                self.n_consecutive_successes += 1
            else:
                self.n_consecutive_successes = 0
            self._tasks_done = False
            wandb.log(
                {
                    "n_successes": self.n_successes,
                    "n_consecutive_successes": self.n_consecutive_successes,
                },
                step=self.model.num_timesteps,
            )
        if (
            self.max_n_consecutive_successes is not None
            and self.n_consecutive_successes >= self.max_n_consecutive_successes
        ):
            return False
        return True

    def _on_rollout_end(self):
        mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
        mean_lenght = np.mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
        current_step = self.model.num_timesteps
        wandb.log(
            {"mean_ep_return": mean_reward, "mean_ep_lenght": mean_lenght},
            step=current_step,
        )


if __name__ == "__main__":
    config = {
        "agent": "MaskablePPO",
        "policy_type": "MlpPolicy",
        "total_timesteps": 2e5,
        "max_n_consecutive_successes": 100,
        "env_name": "RandomCrafting-v1",
        "max_episode_steps": 50,
        "n_items": 50,
        "n_tools": 0,
        "n_foundables": 5,
        "n_zones": 1,
        "task": "random_item",
    }

    def make_env():
        # env = gym.make(config["env_name"])
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
        else:
            raise ValueError
        env = Monitor(env)  # record stats such as returns
        return env

    run = wandb.init(
        project="crafting-benchmark",
        config=config,
        monitor_gym=True,  # auto-upload the videos of agents playing the game
    )
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

    # Draw requirement graph
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    crafting_env.world.draw_requirements_graph(ax)
    plt.title(str(task))
    plt.savefig("requirement_graph.jpg")

    # Draw solving option
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)

    all_options = crafting_env.world.get_all_options()
    all_options_list = list(all_options.values())
    used_nodes_all = nodes_histograms(all_options_list)
    solving_option: Option = all_options[f"Get {task.goal_item}"]
    solving_option.graph.unrolled_graph.draw(ax)

    plt.title(str(solving_option))
    plt.savefig("solving_option.jpg")

    # Compute complexities
    lcomp, comp_saved = learning_complexity(solving_option, used_nodes_all)
    print(f"{str(solving_option)}: {lcomp} ({comp_saved})")
    for node in solving_option.graph.unrolled_graph.nodes():
        print(node, node.complexity)

    wandb.log(
        {
            "task": str(task),
            "solving_option": str(solving_option),
            "learning_complexity": lcomp,
            "total_complexity": lcomp + comp_saved,
            "saved_complexity": comp_saved,
            "requirement_graph": wandb.Image("requirement_graph.jpg"),
            "solving_option": wandb.Image("solving_option.jpg"),
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
