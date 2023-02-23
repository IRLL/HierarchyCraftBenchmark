import gym
import wandb

from crafting.examples import MineCraftingEnv

from craftbench.wandbench import WandbCallback
from craftbench.make_env import record_wrap_env
from craftbench.make_agent import load_agent

PROJECT = "minecrafting-v1-benchmark"

DEFAULT_CONFIG = {
    "agent": "MaskablePPO",
    "agent_seed": 0,
    "policy_type": "MlpPolicy",
    "pi_n_layers": 3,
    "pi_units_per_layer": 64,
    "vf_n_layers": 3,
    "vf_units_per_layer": 64,
    "total_timesteps": 1e6,
    "max_n_consecutive_successes": 200,
    "env_name": "MineCrafting-Platinium-v1",
    "max_step": 1000,
    "record_videos": False,
}


def benchmark_mskppo():
    run = wandb.init(project=PROJECT, config=DEFAULT_CONFIG, monitor_gym=True)
    config = wandb.config
    params_logs = {}

    # Build env
    max_step = config["max_step"]
    crafting_env: MineCraftingEnv = gym.make(config["env_name"], max_step=max_step)
    if config.get("record_videos", False):
        video_path = f"videos/{run.id}"
        env = record_wrap_env(env, video_path)
    else:
        env = crafting_env
    params_logs["purpose"] = str(crafting_env.purpose)

    # Build neural networks architecture from config
    pi_arch = [config["pi_units_per_layer"] for _ in range(config["pi_n_layers"])]
    vf_arch = [config["vf_units_per_layer"] for _ in range(config["vf_n_layers"])]

    # Build agent
    agent = load_agent(
        agent_name=config["agent"],
        env=env,
        policy_type=config["policy_type"],
        net_arch={"pi": pi_arch, "vf": vf_arch},
        seed=config["agent_seed"],
    )

    wandb.log(params_logs)
    agent.learn(
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(
            verbose=2,
            infos_log_freq=max_step if max_step is not None else 500,
            max_n_consecutive_successes=config["max_n_consecutive_successes"],
        ),
        progress_bar=True,
    )

    run.finish()


if __name__ == "__main__":
    benchmark_mskppo()
