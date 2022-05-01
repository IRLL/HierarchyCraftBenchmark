from typing import List
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


def plot_regression(df: pd.DataFrame, x_name: str, y_name: str, ax: plt.Axes, deg=1):
    experiment_settings = ["env_seed", "task_seed"]

    success_df = df[~df[x_name].isna()]
    fail_df = df[df[x_name].isna()]

    mean_succ_df, uncertainty_succ_df = get_mean_and_uncertainty_by_groups(
        success_df, experiment_settings
    )

    mean_fail_df, uncertainty_fail_df = get_mean_and_uncertainty_by_groups(
        fail_df, experiment_settings
    )

    errorbar_config = {
        "marker": "",
        "markersize": 2,
        "linestyle": "",
        "capsize": 2,
        "elinewidth": 1,
    }

    step = mean_succ_df[x_name]
    tcomp = mean_succ_df[y_name]

    ax.errorbar(
        tcomp,
        step,
        yerr=uncertainty_succ_df[x_name],
        color="b",
        label="Success",
        **errorbar_config,
    )

    stepfail = mean_fail_df["_step"]
    tcompfail = mean_fail_df[y_name]
    ax.errorbar(
        tcompfail,
        stepfail,
        yerr=uncertainty_fail_df["_step"],
        color="r",
        label="Fail (step limit)",
        **errorbar_config,
    )

    # Loglog regression on points where multiple runs are sampled
    x = np.log(tcomp[uncertainty_succ_df[x_name] > 0])
    y = np.log(step[uncertainty_succ_df[x_name] > 0])

    # reg: Polynomial = Polynomial([0]).fit(x, y, deg=deg)
    # x_reg, y_reg = reg.linspace(domain=[np.min(np.log(tcomp)), np.max(np.log(15000))])
    # ax.plot(
    #     np.exp(x_reg),
    #     np.exp(y_reg),
    #     color="g",
    #     label=f"Polyfit: {np.array_str(reg.coef, precision=3, suppress_small=True)}",
    # )

    reg = linregress(x, y)
    x_reg = np.linspace(np.min(np.log(tcomp)), np.max(np.log(15000)), 100)
    y_reg = reg.slope * x_reg + reg.intercept
    ax.plot(
        np.exp(x_reg),
        np.exp(y_reg),
        color="g",
        label=f"{reg.slope:.3f}x + {reg.intercept:.3f}: r={reg.rvalue:.3f}",
    )

    ax.loglog()
    ax.set_ylim([np.min(step), 1.1e6])
    ax.set_xlim([np.min(tcomp), 15000])
    ax.legend(loc="upper left", fontsize=5)
    return reg


if __name__ == "__main__":
    experiments_df = pd.read_csv("runs_data.csv")
    complexity_name = "total_complexity"

    # Replace invalid task_seed
    experiments_df["task_seed"].mask(
        experiments_df["task_seed"] == "[0]", 0, inplace=True
    )

    # Gather grid values
    pi_units_per_layer_values = pd.unique(experiments_df["pi_units_per_layer"])
    pi_units_per_layer_values.sort()

    vf_units_per_layer_values = pd.unique(experiments_df["vf_units_per_layer"])
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
            reg = plot_regression(
                filtered_df,
                x_name="csuccess50_step",
                y_name=complexity_name,
                deg=deg,
                ax=ax,
            )

            print(f"pi={pi_units}, vf={vf_units}: {reg}")
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
        f"Correlation between steps to convergence and {complexity_name}"
        " for multiple network sizes",
        fontsize=16,
    )
    fig.text(s="Units per layer in value network", x=0.5, y=1 - 0.08, ha="center")
    fig.supylabel("Units per layer in policy network", x=0.08)

    fig.supxlabel(f"{complexity_name.capitalize()}", y=0.04)
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
