import os
import gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.a2c.a2c import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy

from sb3_contrib.ppo_mask.ppo_mask import MaskablePPO

from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback

from crafting import MineCraftingEnv

from gym.envs.classic_control import CartPoleEnv


class WandbCallback(WandbCallback):
    def _on_rollout_end(self):
        mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
        mean_lenght = np.mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
        current_step = self.model.num_timesteps
        wandb.log(
            {
                "mean_ep_return": mean_reward,
                "mean_ep_lenght": mean_lenght,
            },
            step=current_step,
        )


config = {
    "agent": "MaskablePPO",
    "policy_type": "MlpPolicy",
    "total_timesteps": 2e5,
    "env_name": "MineCrafting-v1",
    "max_episode_steps": 100,
}
run = wandb.init(
    project="minecrafting-benchmark",
    config=config,
    monitor_gym=True,  # auto-upload the videos of agents playing the game
)
config = wandb.config


def make_env():
    # env = gym.make(config["env_name"])
    env = MineCraftingEnv(
        tasks=["obtain_enchanting_table"],
        tasks_can_end=[True],
        max_step=config["max_episode_steps"],
    )
    env = Monitor(env)  # record stats such as returns
    return env


env = DummyVecEnv([make_env])
env = VecVideoRecorder(
    env,
    f"videos/{run.id}",
    record_video_trigger=lambda step: step % 10000 == 0,
    video_length=200,
)

# net_arch = [dict(pi=[64, 64], vf=[64, 64])]
# ActorCriticPolicy()
agent = eval(config["agent"])(config["policy_type"], env, verbose=1)

agent.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        verbose=2,
    ),
)
run.finish()

#
# obs = env.reset()
# done = False
# total_return = 0

# while not done:
#     env.render(mode='rgb_array')
#     legal_actions = np.arange(len(env.get_action_is_legal()))[env.get_action_is_legal()]
#     action = np.random.choice(legal_actions)
#     obs, reward, done, _ = env.step(action)
#     total_return += reward

# print(total_return)
