from typing import Any, Dict

import pandas as pd
import wandb
from tqdm import tqdm


def is_relevent_parameter(name: str):
    return (
        name in ("task_name", "_step")
        or name.startswith("pi")
        or name.startswith("vf")
        or name.startswith("mean_ep")
        or name.startswith("n_")
        or name.endswith("complexity")
        or name.endswith("seed")
    )


def add_to_dict_of_lists(dictionary: Dict[Any, list], new_dict: Dict[Any, list]):
    for name, item in new_dict.items():
        if is_relevent_parameter(name):
            try:
                dictionary[name].append(item)
            except KeyError:
                dictionary[name] = [item]
    for key, dict_list in dictionary.items():
        if key not in new_dict:
            dict_list.append(None)


if __name__ == "__main__":
    api = wandb.Api()
    entity, project = "mathisfederico", "crafting-benchmark"
    runs = api.runs(entity + "/" + project)

    summary_dict, config_dict = {}, {}
    name_list, sweep_list = [], []
    succ100_step_list, csucc100_step_list, csucc50_step_list = [], [], []

    loader = tqdm(runs, total=len(runs))
    for run in loader:
        if (
            run.state == "finished"
            and run.sweep is not None
            and run.sweep.name
            in (
                "ch1poicp",
                "gdopl9qq",
                "ec4ezbfn",
                "pnwmf2am",
            )
        ):
            # .summary contains the output keys/values for metrics like accuracy.
            #  We call ._json_dict to omit large files
            add_to_dict_of_lists(summary_dict, run.summary._json_dict)

            # .config contains the hyperparameters.
            add_to_dict_of_lists(config_dict, run.config)

            # .name is the human-readable name of the run.
            name_list.append(run.name)
            sweep_list.append(run.sweep.name)

            hist_df = run.history()
            succ100_step = hist_df[hist_df["n_successes"] >= 100]["_step"].min()
            csucc50_step = hist_df[hist_df["n_consecutive_successes"] >= 50][
                "_step"
            ].min()
            csucc100_step = hist_df[hist_df["n_consecutive_successes"] >= 100][
                "_step"
            ].min()

            succ100_step_list.append(succ100_step)
            csucc50_step_list.append(csucc50_step)
            csucc100_step_list.append(csucc100_step)

            tcomp = run.summary._json_dict["total_complexity"]
            scomp = run.summary._json_dict["saved_complexity"]
            pi_units = run.config["pi_units_per_layer"]
            vf_units = run.config["vf_units_per_layer"]
            loader.set_description(
                f"{run.name: <25} | {tcomp}({scomp}) | pi={pi_units: <4}, vf={vf_units: <4} | "
                f"{succ100_step} {csucc50_step} {csucc100_step}"
            )

    runs_df = pd.DataFrame(
        {
            "name": name_list,
            "sweep": sweep_list,
            "success100_step": succ100_step_list,
            "csuccess100_step": csucc100_step_list,
            "csuccess50_step": csucc50_step_list,
            **summary_dict,
            **config_dict,
        }
    )

    print(list(summary_dict.keys()), list(config_dict.keys()))
    print(runs_df.head(10))

    runs_df.to_csv("project.csv")
