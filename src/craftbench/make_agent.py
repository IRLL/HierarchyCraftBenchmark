from typing import Dict, List

from gym import Env
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.base_class import BaseAlgorithm
from sb3_contrib import MaskablePPO

NAME_TO_AGENT = {
    "PPO": PPO,
    "DQN": DQN,
    "A2C": A2C,
    "MaskablePPO": MaskablePPO,
}


def load_agent(
    agent_name: str,
    env: Env,
    policy_type: str,
    net_arch: Dict[List[int]],
    seed: int,
) -> BaseAlgorithm:
    """Load a sb3 agent from it's name."""
    agent_class = NAME_TO_AGENT[agent_name]
    if agent_name == "DQN":
        net_arch = net_arch["pi"]
    return agent_class(
        policy=policy_type,
        env=env,
        policy_kwargs={"net_arch": net_arch},
        seed=seed,
        verbose=1,
    )
