import logging

from unified_planning.shortcuts import get_env


from hcraft.examples import MINICRAFT_GYM_ENVS, MINEHCRAFT_GYM_ENVS


import gym
from craftbench.planning import benchmark_planning


def main():
    for env_id in MINICRAFT_GYM_ENVS + MINEHCRAFT_GYM_ENVS[2:32]:
        env = gym.make(env_id, max_step=500)
        benchmark_planning(env, experiment_name=env_id, force_experiment=True)


if __name__ == "__main__":
    get_env().credits_stream = None
    logging.basicConfig(level=logging.INFO)
    main()
