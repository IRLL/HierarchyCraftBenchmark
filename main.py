import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback

from crafting import MineCraftingEnv

from gym.envs.classic_control import CartPoleEnv

config = {
    "agent": "PPO",
    "policy_type": "MlpPolicy",
    "total_timesteps": 200000,
    "env_name": "MineCrafting-v1",
    "max_episode_steps": 100,
}
run = wandb.init(
    project="sb3",
    config=config,
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)

def make_env():
    # env = gym.make(config["env_name"])
    env = MineCraftingEnv(tasks=['obtain_enchanting_table'], max_step=config['max_episode_steps'])
    env = Monitor(env)  # record stats such as returns
    return env

env = DummyVecEnv([make_env])
env = VecVideoRecorder(env, f"videos/{run.id}",
    record_video_trigger=lambda step: step % 10000 == 0, video_length=200)

agent = PPO(config["policy_type"], env, verbose=1)

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
