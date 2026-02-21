"""
visualization_utils.py — Shared plotting functions and style presets.

Provides ready-made chart generators used by person modules and integration:
  - bar_chart()       — grouped or simple bar chart
  - radar_chart()     — multi-dimension radar / spider chart
  - heatmap()         — annotated 2D heatmap
  - line_chart()      — line plot with optional error bands
  - save_figure()     — save with tight layout and high DPI

All functions return a matplotlib Figure so callers can further customize.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path

matplotlib.use("Agg")  # non-interactive backend

# ── Style Presets ───────────────────────────────────────────────────────────

STYLE = {
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
}

MODEL_COLORS = {
    "gpt2":    "#4C72B0",
    "llama3":  "#DD8452",
    "flan_t5": "#55A868",
}

plt.rcParams.update(STYLE)
sns.set_palette(list(MODEL_COLORS.values()))


# ── Save Helper ─────────────────────────────────────────────────────────────

def save_figure(fig: plt.Figure, path: str | Path, close: bool = True):
    """Save a figure to disk with tight layout."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    if close:
        plt.close(fig)
    print(f"  ✓ Saved plot → {path.name}")


# ── Bar Chart ───────────────────────────────────────────────────────────────

def bar_chart(
    data: dict[str, list[float]],
    labels: list[str],
    title: str = "",
    ylabel: str = "Score",
    ylim: tuple = (0, 1),
) -> plt.Figure:
    """
    Grouped bar chart.

    Args:
        data:   {model_name: [val1, val2, ...]}
        labels: x-axis category labels
        title:  chart title
        ylabel: y-axis label
        ylim:   y-axis limits

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots()
    x = np.arange(len(labels))
    n_groups = len(data)
    width = 0.8 / n_groups

    for i, (model, values) in enumerate(data.items()):
        color = MODEL_COLORS.get(model, None)
        ax.bar(x + i * width, values, width, label=model, color=color)

    ax.set_xticks(x + width * (n_groups - 1) / 2)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


# ── Radar Chart ─────────────────────────────────────────────────────────────

def radar_chart(
    data: dict[str, list[float]],
    categories: list[str],
    title: str = "Model Comparison",
) -> plt.Figure:
    """
    Radar / spider chart comparing models across multiple dimensions.

    Args:
        data:       {model_name: [score_dim1, score_dim2, ...]}
        categories: dimension names
        title:      chart title
    """
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(subplot_kw={"polar": True})

    for model, values in data.items():
        vals = values + values[:1]
        color = MODEL_COLORS.get(model, None)
        ax.plot(angles, vals, linewidth=2, label=model, color=color)
        ax.fill(angles, vals, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title(title, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    return fig


# ── Heatmap ─────────────────────────────────────────────────────────────────

def heatmap(
    matrix: list[list[float]],
    xlabels: list[str],
    ylabels: list[str],
    title: str = "",
    cmap: str = "YlOrRd",
    annot: bool = True,
    fmt: str = ".2f",
) -> plt.Figure:
    """Annotated 2D heatmap."""
    fig, ax = plt.subplots()
    sns.heatmap(
        np.array(matrix),
        xticklabels=xlabels,
        yticklabels=ylabels,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ── Line Chart ──────────────────────────────────────────────────────────────

def line_chart(
    data: dict[str, tuple[list, list]],
    xlabel: str = "X",
    ylabel: str = "Y",
    title: str = "",
) -> plt.Figure:
    """
    Line chart with optional data series.

    Args:
        data: {model_name: (x_values, y_values)}
    """
    fig, ax = plt.subplots()

    for model, (xs, ys) in data.items():
        color = MODEL_COLORS.get(model, None)
        ax.plot(xs, ys, marker="o", linewidth=2, label=model, color=color)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
