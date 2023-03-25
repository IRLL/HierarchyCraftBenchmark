import gym
import wandb

from hcraft.examples import MineHcraftEnv

from craftbench.wandbench import WandbCallback
from craftbench.make_env import record_wrap_env
from craftbench.make_agent import load_agent

PROJECT = "minehcraft-v1-benchmark"

DEFAULT_CONFIG = {
    "agent": "MaskablePPO",
    "agent_seed": 0,
    "policy_type": "MlpPolicy",
    "pi_n_layers": 3,
    "pi_units_per_layer": 64,
    "vf_n_layers": 3,
    "vf_units_per_layer": 64,
    "invalid_reward": -10,
    "total_timesteps": 1e6,
    "max_n_consecutive_successes": 200,
    "env_name": "MineHCraft-v1",
    "max_step": 0,
    "record_videos": False,
    "device": "cuda",
}


def benchmark_mskppo():
    run = wandb.init(project=PROJECT, config=DEFAULT_CONFIG, monitor_gym=True)
    config = wandb.config
    params_logs = {}

    # Build env
    max_step = config["max_step"] if config["max_step"] > 0 else None
    hcraft_env: MineHcraftEnv = gym.make(
        config["env_name"],
        max_step=max_step,
        invalid_reward=config["invalid_reward"],
    )
    if config.get("record_videos", False):
        video_path = f"videos/{run.id}"
        env = record_wrap_env(env, video_path)
    else:
        env = hcraft_env
    params_logs["purpose"] = str(hcraft_env.purpose)

    # Build neural networks architecture from config
    net_arch = _build_network_architecture(
        config["pi_units_per_layer"],
        config["pi_n_layers"],
        config["vf_units_per_layer"],
        config["vf_n_layers"],
    )

    # Build agent
    agent = load_agent(
        agent_name=config["agent"],
        env=env,
        policy_type=config["policy_type"],
        net_arch=net_arch,
        device=config["device"],
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


def _build_network_architecture(
    pi_units_per_layer: int,
    pi_n_layers: int,
    vf_units_per_layer: int,
    vf_n_layers: int,
):
    pi_arch = [pi_units_per_layer for _ in range(pi_n_layers)]
    vf_arch = [vf_units_per_layer for _ in range(vf_n_layers)]
    return {"pi": pi_arch, "vf": vf_arch}


if __name__ == "__main__":
    benchmark_mskppo()
