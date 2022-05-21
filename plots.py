import os
import sys
from typing import Tuple

import matplotlib.pyplot as plt

from crafting.world.world import World

from option_graph.option import Option
from option_graph.graph import compute_levels


def save_plot(filename: str, dirname: str, title: str, dpi: float):
    """Save a plot to a given dirname with a given filename.

    Create directory if needed.

    """
    plt.title(title)
    dirpath = os.path.join("images", dirname)
    os.makedirs(dirpath, exist_ok=True)
    path = os.path.join(dirpath, filename)
    plt.savefig(path, dpi=dpi)
    return path


def save_requirement_graph(
    name: str,
    world: World,
    title: str,
    figsize: Tuple[float],
    dpi: float = 240,
) -> str:
    """Draw & save requirement graph."""
    _, ax = plt.subplots(figsize=figsize)
    world.draw_requirements_graph(ax)
    return save_plot("requirement_graph.jpg", name, title, dpi)


def save_option_graph(
    option: Option,
    name: str,
    dpi: float = 90,
    unrolled: bool = True,
) -> str:
    """Draw & save an option graph."""
    graph = option.graph
    if unrolled:
        graph = graph.unrolled_graph
    compute_levels(graph)
    width = max(len(nodes) for nodes in graph.graph["nodes_by_level"].values())
    _, ax = plt.subplots(figsize=(max(8, 16 * width / 8), max(4.5, 9 * width / 8)))
    graph.draw(ax, draw_options_hulls=True)

    return save_plot("solving_option.jpg", name, str(option), dpi)
