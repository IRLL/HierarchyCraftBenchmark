import time


from sb3_contrib.ppo_mask.ppo_mask import MaskablePPO

from hebg.metrics.complexity import learning_complexity
from hebg.metrics.complexity.histograms import nodes_histograms
from hebg.behavior import Behavior

import wandb

from crafting.env import CraftingEnv
from crafting.task import RewardShaping, TaskObtainItem

from craftbench.wandbench import WandbCallback
from craftbench.plots import save_requirement_graph, save_heb_graph

from craftbench.make_env import make_env, record_wrap_env

PROJECT = "stackedcrafting-benchmark"

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
    "env_name": "TowerCrafting-v1",
    "reward_shaping": RewardShaping.NONE.value,
    "height": 2,
    "width": 4,
}


def run_solve(env: CraftingEnv, solver: Behavior) -> int:
    """Count how much steps are needed for the solver to finish.

    Args:
        env (CraftingEnv): The Crafting environment containing an finishing task.
        solver (Option): The solver to test the lenght of.

    Returns:
        int: Number of steps needed for the solver to complete the task.
    """
    step = 0
    done = False
    observation = env.reset()
    while not done:
        action = solver(observation)
        observation, _, done, _ = env.step(action)
        step += 1

    assert step < env.max_step
    return step


def benchmark_mskppo(
    save_req_graph: bool = False,
    save_sol_graph: bool = False,
):
    run = wandb.init(project=PROJECT, config=DEFAULT_CONFIG, monitor_gym=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dirname = f"{timestamp}-{run.id}"
    config = wandb.config
    params_logs = {}

    # Build env
    crafting_env = make_env(config)
    task: TaskObtainItem = crafting_env.tasks[0]  # Assume only one task
    params_logs["task"] = str(task)
    params_logs["_env_name"] = crafting_env.name

    # Get solving option
    all_options = crafting_env.world.get_all_options()
    all_options_list = list(all_options.values())
    solving_option: Behavior = all_options[f"Get {task.goal_item}"]
    params_logs["solving_option"] = str(solving_option)

    # Adapt max_step to solving option size
    steps_to_solve = run_solve(crafting_env, solving_option)
    # crafting_env.max_step = int(4 * steps_to_solve)  # Give a 300% margin error
    crafting_env.max_step = 2**crafting_env.world.n_items
    params_logs["steps_to_solve"] = steps_to_solve
    params_logs["_max_step"] = crafting_env.max_step

    # Save goal solving graph
    if save_sol_graph:
        solving_graph = solving_option.graph.unrolled_graph
        solving_heb_graph_path = save_heb_graph(solving_graph, run_dirname)
        params_logs["solving_heb_graph"] = wandb.Image(solving_heb_graph_path)

    # Compute complexities
    used_nodes_all = nodes_histograms(all_options_list)
    lcomp, comp_saved = learning_complexity(solving_option, used_nodes_all)
    print(
        f"OPTION: {str(solving_option)}:"
        f"Complexities total={lcomp + comp_saved},"
        f" saved={comp_saved}, learn={lcomp}"
    )
    params_logs.update(
        {
            "learning_complexity": lcomp,
            "total_complexity": lcomp + comp_saved,
            "saved_complexity": comp_saved,
        }
    )

    # Get & save requirements graph
    if save_req_graph:
        requirement_graph_path = save_requirement_graph(
            run_dirname,
            crafting_env.world,
            title=str(crafting_env.world),
            figsize=(32, 18),
        )
        params_logs["requirement_graph"] = wandb.Image(requirement_graph_path)

    # Add video recording
    video_path = f"videos/{run.id}"
    env = record_wrap_env(crafting_env, video_path)

    # Build neural networks architecture from config
    pi_arch = [config["pi_units_per_layer"] for _ in range(config["pi_n_layers"])]
    vf_arch = [config["vf_units_per_layer"] for _ in range(config["vf_n_layers"])]
    net_arch = [dict(pi=pi_arch, vf=vf_arch)]

    # Build agent
    agent = MaskablePPO(
        config["policy_type"],
        env,
        policy_kwargs={"net_arch": net_arch},
        seed=config["agent_seed"],
        verbose=1,
    )

    wandb.log(params_logs)

    # pylint: disable=unexpected-keyword-arg
    agent.learn(
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(
            verbose=2,
            max_n_consecutive_successes=config["max_n_consecutive_successes"],
        ),
    )

    run.finish()


if __name__ == "__main__":
    benchmark_mskppo(save_req_graph=True, save_sol_graph=False)
