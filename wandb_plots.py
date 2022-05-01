from typing import List, Optional
import pandas as pd

import numpy as np
from scipy.stats import linregress

import matplotlib.pyplot as plt


def get_mean_and_uncertainty_by_groups(df: pd.DataFrame, groups: List[str]):
    mean_df = df.groupby(groups).mean()
    std_df = df.groupby(groups).std()
    std_df.fillna(0.0, inplace=True)
    count_df = df.groupby(groups).count()
    uncertainty_df = 2 * std_df / np.sqrt(count_df)
    return mean_df, uncertainty_df


def plot_linregress(
    df: pd.DataFrame,
    x_name: str,
    y_name: str,
    ax: plt.Axes,
    fail_y_name: str = "_step",
    groups: Optional[List[str]] = None,
):
    success_df = df[~df[y_name].isna()]
    fail_df = df[df[y_name].isna()]

    mean_succ_df, uncertainty_succ_df = get_mean_and_uncertainty_by_groups(
        success_df, groups
    )

    mean_fail_df, uncertainty_fail_df = get_mean_and_uncertainty_by_groups(
        fail_df, groups
    )

    errorbar_config = {
        "marker": "",
        "markersize": 2,
        "linestyle": "",
        "capsize": 2,
        "elinewidth": 1,
    }

    mean_x = mean_succ_df[x_name]
    mean_y = mean_succ_df[y_name]
    ax.errorbar(
        mean_x,
        mean_y,
        yerr=uncertainty_succ_df[y_name],
        color="b",
        label="Success",
        **errorbar_config,
    )

    mean_x_fail = mean_fail_df[x_name]
    mean_y_fail = mean_fail_df[fail_y_name]
    ax.errorbar(
        mean_x_fail,
        mean_y_fail,
        yerr=uncertainty_fail_df[fail_y_name],
        color="r",
        label="Fail (step limit)",
        **errorbar_config,
    )

    min_x = df[x_name].min()
    max_x = df[x_name].max()

    min_y = df[fail_y_name].min()
    max_y = df[fail_y_name].max()

    # Loglog regression on points where multiple runs are sampled
    multi_sampled = uncertainty_succ_df[y_name] > 0
    x = np.log(mean_x[multi_sampled])
    y = np.log(mean_y[multi_sampled])

    reg = linregress(x, y)
    x_reg = np.linspace(np.log(min_x), np.log(max_x) * 1.05, 100)
    y_reg = reg.slope * x_reg + reg.intercept
    ax.plot(
        np.exp(x_reg),
        np.exp(y_reg),
        color="g",
        label=f"{reg.slope:.3f}x + {reg.intercept:.3f}: r={reg.rvalue:.3f}",
    )

    ax.loglog()
    ax.set_xlim([min_x, 1.1 * max_x])
    ax.set_ylim([min_y, 1.1 * max_y])
    ax.legend(loc="upper left")
    return reg


def plot_grid(runs_df: pd.DataFrame, x_name: str, y_name: str):
    # Gather grid values
    pi_units_per_layer_values = pd.unique(runs_df["pi_units_per_layer"])
    pi_units_per_layer_values.sort()

    vf_units_per_layer_values = pd.unique(runs_df["vf_units_per_layer"])
    vf_units_per_layer_values.sort()

    # Prepare subplots
    fig, axes = plt.subplots(
        len(pi_units_per_layer_values),
        len(vf_units_per_layer_values),
        sharex=True,
        sharey=True,
    )

    deg = 1
    coefs = np.zeros(axes.shape + (deg + 1,))
    for i, pi_units in enumerate(pi_units_per_layer_values):
        for j, vf_units in enumerate(vf_units_per_layer_values):
            filtered_df = experiments_df[
                (experiments_df["vf_units_per_layer"] == vf_units)
                & (experiments_df["pi_units_per_layer"] == pi_units)
            ]
            ax = axes[i, j]
            reg = plot_linregress(
                filtered_df,
                x_name=x_name,
                y_name=y_name,
                ax=ax,
                groups=["env_seed", "task_seed"],
            )

            # coefs[i, j, :] = np.array(reg.coef)
            coefs[i, j, :] = np.array([reg.intercept, reg.slope])
            # axes[i, j].set_suptitle(f"{reg.slope} {reg.rvalue}")
            labelright = False
            if i == 0:
                ax.set_title(vf_units)
            if j == 0:
                ax.set_ylabel(pi_units)
            if j == len(vf_units_per_layer_values) - 1:
                labelright = True
            ax.tick_params(
                axis="y",
                which="both",
                labelleft=False,
                left=False,
                right=labelright,
                labelright=labelright,
                direction="out",
            )
            # axes[i, j].setTitle(f"pi-{pi_units} vf-{vf_units}")

    fig.suptitle(
        f"Correlation between steps to convergence and {x_name}"
        " for multiple network sizes",
        fontsize=16,
    )
    fig.text(s="Units per layer in value network", x=0.5, y=1 - 0.08, ha="center")
    fig.supylabel("Units per layer in policy network", x=0.08)

    fig.supxlabel(f"{x_name.capitalize()}", y=0.04)
    fig.text(s="Steps", x=1 - 0.08, y=0.5, rotation=270, va="center")
    plt.show()

    fig, axes = plt.subplots(1, deg + 1, sharey=True)
    X, Y = np.meshgrid(pi_units_per_layer_values, vf_units_per_layer_values)

    for degree in range(deg + 1):
        ax = axes[degree]
        coefs_deg = coefs[:, :, degree]
        im = ax.imshow(coefs_deg, cmap="copper")

        # Show all ticks and label them with the respective list entries
        ax.set_xlabel("Units per layer in policy network")
        ax.set_xticks(
            np.arange(len(pi_units_per_layer_values)),
            labels=pi_units_per_layer_values,
        )

        if degree == 0:
            ax.set_ylabel("Units per layer in value network")
            ax.set_yticks(
                np.arange(len(vf_units_per_layer_values)),
                labels=vf_units_per_layer_values,
            )

        # Loop over data dimensions and create text annotations.
        for i in range(len(pi_units_per_layer_values)):
            for j in range(len(vf_units_per_layer_values)):
                text = ax.text(
                    j,
                    i,
                    f"{coefs_deg[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="w",
                    fontsize=9,
                )

        ax.set_title(f"Degree {degree}")

    fig.suptitle("Polynomial fit coefficients for different network sizes")
    plt.show()


def plot_single_linregress(experiments_df: pd.DataFrame, x_name: str, y_name: str):
    ax = plt.subplot()
    plot_linregress(
        experiments_df,
        x_name=x_name,
        y_name=y_name,
        ax=ax,
        groups=["env_seed", "task_seed"],
    )
    pretty_x_name = x_name.replace("_", " ")
    plt.title(f"Steps to 50 consecutive successes with respect to {pretty_x_name}")
    plt.xlabel(f"{pretty_x_name.capitalize()}")

    pretty_y_name_parts = ["Steps to reach"]
    pretty_y_name_parts.append(y_name.split("_")[0][-2:])
    if y_name.startswith("c"):
        pretty_y_name_parts.append("consecutive")
    pretty_y_name_parts.append("successes")
    pretty_y_name = " ".join(pretty_y_name_parts)
    plt.ylabel(f"{pretty_y_name}")
    plt.show()


if __name__ == "__main__":
    experiments_df = pd.read_csv("runs_data_64.csv")

    # Replace invalid task_seed
    experiments_df["task_seed"].mask(
        experiments_df["task_seed"] == "[0]", 0, inplace=True
    )

    # plot_grid(experiments_df, "total_complexity", "csuccess50_step")
    filtered_df = experiments_df[
        (experiments_df["vf_units_per_layer"] == 64)
        & (experiments_df["pi_units_per_layer"] == 64)
        & (experiments_df["reward_shaping"] == 2)
    ]
    plot_single_linregress(filtered_df, "learning_complexity", "csuccess50_step")
    plot_single_linregress(filtered_df, "total_complexity", "csuccess50_step")

    x_range = np.linspace(
        np.min(filtered_df["learning_complexity"]),
        np.max(filtered_df["total_complexity"]),
    )
    plt.plot(x_range, x_range, label="Indentity", linestyle=":", color="g")
    plt.scatter(
        filtered_df["total_complexity"],
        filtered_df["learning_complexity"],
        marker="+",
        label="experiments",
    )
    plt.legend()
    plt.xlabel("Total complexity")
    plt.ylabel("Learning complexity")
    plt.loglog()
    plt.title("Correlation between complexities")
    plt.show()
