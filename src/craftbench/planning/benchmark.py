import os
from pathlib import Path
import json

from typing import List, Dict, Optional
import logging

from unified_planning.shortcuts import get_env

from hcraft.task import GetItemTask
from hcraft.requirements import RequirementNode, req_node_name
from hcraft.env import HcraftEnv
from hcraft.examples import HCRAFT_GYM_ENVS

from tqdm import trange
import gym

PLANNING_METRICS = [
    "States Evaluated",
    "Planning Time (msec)",
    "Heuristic Time (msec)",
    "Search Time (msec)",
    "Plan-Length",
]


def benchmark_planning(
    env: HcraftEnv,
    experiment_name: Optional[str] = None,
    n_samples: int = 10,
    save_path: Path = Path("experiments/planning"),
    force_experiment: bool = False,
) -> bool:
    experiment_name = experiment_name if experiment_name is not None else env.name
    os.makedirs(save_path, exist_ok=True)
    file_path = Path(save_path) / f"{experiment_name}.json"
    if file_path.exists() and not force_experiment:
        return True

    metrics: Dict[str, List[float]] = {}
    if env.purpose.tasks and isinstance(env.purpose.tasks[0], GetItemTask):
        goal_item = env.purpose.tasks[0].item_stack.item
        req_node = req_node_name(goal_item, RequirementNode.ITEM)

        metrics["Requirements level"] = env.world.requirements.graph.nodes[req_node][
            "level"
        ]
    for metric in PLANNING_METRICS:
        metrics[metric] = []

    for _ in trange(n_samples, desc=experiment_name):
        env.reset()
        done = False
        problem = env.planning_problem(timeout=60)
        while not done:
            try:
                action = problem.action_from_plan(env.state)
            except ValueError:
                logging.info(f"Failed to find a plan for {experiment_name}")
                return False
            if action is None:
                logging.info(f"No action to take for {experiment_name}")
                return False
            _observation, _reward, done, _ = env.step(action)
        assert env.purpose.terminated, f"Plan failed :{problem.plans}"
        assert len(problem.stats) == 1
        stats = problem.stats[0]
        for metric in PLANNING_METRICS:
            metrics[metric].append(stats[metric])

    if file_path.exists():
        os.remove(file_path)
    with open(file_path, "w") as save_file:
        json.dump(metrics, save_file, indent=4)
        logging.info(f"Saved experiment at {file_path}")

    return True


def main():
    for env_id in HCRAFT_GYM_ENVS[2:]:
        env = gym.make(env_id, max_step=500)
        benchmark_planning(env, experiment_name=env_id, force_experiment=True)


if __name__ == "__main__":
    get_env().credits_stream = None
    logging.basicConfig(level=logging.INFO)
    main()
