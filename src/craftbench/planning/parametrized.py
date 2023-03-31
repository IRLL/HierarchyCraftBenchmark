import logging

from unified_planning.shortcuts import get_env


from hcraft.examples import TowerHcraftEnv, RecursiveHcraftEnv, LightRecursiveHcraftEnv

from craftbench.planning import benchmark_planning


def main():
    for height in range(1, 9):
        for width in range(1, 5):
            if width > 1 and height > 6:
                continue
            if width > 2 and height > 4:
                continue
            if width == 4 and height == 4:
                continue
            env = TowerHcraftEnv(height=height, width=width, max_step=1000)
            benchmark_planning(env, experiment_name=env.name, force_experiment=False)

    for n_items in range(1, 11):
        env = RecursiveHcraftEnv(n_items=n_items, max_step=1500)
        benchmark_planning(env, experiment_name=env.name, force_experiment=False)

    for n_items in range(1, 12):
        for n_req_items in range(2, n_items + 1):
            env = LightRecursiveHcraftEnv(
                n_items=n_items, n_required_previous=n_req_items, max_step=1500
            )
            benchmark_planning(env, experiment_name=env.name, force_experiment=False)


if __name__ == "__main__":
    get_env().credits_stream = None
    logging.basicConfig(level=logging.INFO)
    main()
