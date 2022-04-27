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
    name_list, sweep_list, state_list = [], [], []

    loader = tqdm(runs, total=len(runs))
    for run in loader:
        if run.sweep is not None and run.sweep.name in (
            "ch1poicp",
            "gdopl9qq",
            "ec4ezbfn",
            "pnwmf2am",
        ):
            loader.set_description(f"{run.name: <25}")
            # .summary contains the output keys/values for metrics like accuracy.
            #  We call ._json_dict to omit large files
            add_to_dict_of_lists(summary_dict, run.summary._json_dict)

            # .config contains the hyperparameters.
            add_to_dict_of_lists(config_dict, run.config)

            # .name is the human-readable name of the run.
            name_list.append(run.name)
            sweep_list.append(run.sweep.name)
            state_list.append(run.state)

    runs_df = pd.DataFrame(
        {
            "name": name_list,
            "sweep": sweep_list,
            "state": state_list,
            **summary_dict,
            **config_dict,
        }
    )

    print(list(summary_dict.keys()), list(config_dict.keys()))
    print(runs_df.head(10))

    runs_df.to_csv("project.csv")
