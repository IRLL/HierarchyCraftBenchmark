import os
from pathlib import Path
import json

import logging
from typing import Optional

from tqdm import trange

from hcraft.task import GetItemTask
from hcraft.requirements import RequirementNode, req_node_name
from hcraft.env import HcraftEnv

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
    experiments_data = []
    experiment_name = experiment_name if experiment_name is not None else env.name
    os.makedirs(save_path, exist_ok=True)
    file_path = Path(save_path) / f"{experiment_name}.json"
    if file_path.exists() and not force_experiment:
        return True

    requirements_level = None
    if env.purpose.tasks and isinstance(env.purpose.tasks[0], GetItemTask):
        goal_item = env.purpose.tasks[0].item_stack.item
        req_node = req_node_name(goal_item, RequirementNode.ITEM)
        requirements_level = env.world.requirements.graph.nodes[req_node]["level"]

    for trial in trange(n_samples, desc=experiment_name):
        metrics = {"env_id": experiment_name, "trial": trial}
        if requirements_level is not None:
            metrics["Requirements Level"] = requirements_level
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
        if not env.purpose.terminated:
            logging.info(f"Plan failed for {experiment_name} :{problem.plans}")
            return False
        assert len(problem.stats) == 1
        stats = problem.stats[0]
        for metric in PLANNING_METRICS:
            small_name = metric.replace("-", " ")
            metrics[small_name] = stats[metric]
        experiments_data.append(metrics)

    if file_path.exists():
        os.remove(file_path)
    with open(file_path, "w") as save_file:
        json.dump(experiments_data, save_file, indent=4)
        logging.info(f"Saved experiment at {file_path}")

    return True
