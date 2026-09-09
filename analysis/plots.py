"""Reusable plotting functions for BO campaign reports."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .pareto import is_non_dominated


def _savefig(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_objective_tradeoff(
    df: pd.DataFrame,
    objective_names: Sequence[str],
    path: str | Path,
    *,
    iteration_column: str = "Iteration",
    candidate_df: pd.DataFrame | None = None,
    objective_directions: Sequence[str] | None = None,
) -> None:
    """Plot objective tradeoff with Pareto points highlighted."""
    if len(objective_names) != 2:
        return
    complete = df.dropna(subset=list(objective_names)).copy()
    if complete.empty:
        return

    mask = is_non_dominated(
        complete[list(objective_names)].to_numpy(dtype=float),
        objective_directions or ["maximize", "maximize"],
    )
    complete["Pareto"] = mask

    plt.figure(figsize=(7, 5))
    hue = iteration_column if iteration_column in complete.columns else None
    sns.scatterplot(
        data=complete,
        x=objective_names[0],
        y=objective_names[1],
        hue=hue,
        style="Pareto",
        s=70,
        edgecolor="black",
        linewidth=0.4,
    )
    if candidate_df is not None:
        candidate_cols = [f"Predicted Mean {name}" for name in objective_names]
        if set(candidate_cols).issubset(candidate_df.columns):
            plt.scatter(
                candidate_df[candidate_cols[0]],
                candidate_df[candidate_cols[1]],
                marker="X",
                s=110,
                c="tab:red",
                edgecolor="black",
                linewidth=0.6,
                label="Proposed candidates",
            )
            plt.legend()
    plt.title("Objective Tradeoff and Pareto Front")
    plt.grid(True, alpha=0.25)
    _savefig(path)


def plot_hypervolume_progress(progress_df: pd.DataFrame, path: str | Path) -> None:
    """Plot hypervolume by iteration."""
    if progress_df.empty or "Hypervolume" not in progress_df.columns:
        return
    plt.figure(figsize=(6.5, 4))
    plt.plot(progress_df["Iteration"], progress_df["Hypervolume"], marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Hypervolume")
    plt.title("Observed Hypervolume Progress")
    plt.grid(True, alpha=0.25)
    _savefig(path)


def plot_design_space_pairplot(
    df: pd.DataFrame,
    parameter_names: Sequence[str],
    path: str | Path,
    *,
    hue: str | None = "Iteration",
) -> None:
    """Plot pairwise design-space coverage."""
    available = [name for name in parameter_names if name in df.columns]
    if len(available) < 2:
        return
    hue_arg = hue if hue in df.columns else None
    plot = sns.pairplot(
        df[available + ([hue_arg] if hue_arg else [])],
        vars=available,
        hue=hue_arg,
        corner=True,
        diag_kind="hist",
        plot_kws={"s": 28, "edgecolor": "black", "linewidth": 0.35, "alpha": 0.75},
    )
    plot.fig.suptitle("Design Space Coverage", y=1.02)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plot.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(plot.fig)


def plot_calibration_curve(
    actual: np.ndarray,
    predicted: np.ndarray,
    std: np.ndarray,
    path: str | Path,
    *,
    title: str = "Predictive Calibration",
) -> None:
    """Plot empirical interval coverage against nominal Gaussian coverage."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    std = np.asarray(std, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted) & np.isfinite(std) & (std > 0)
    if valid.sum() == 0:
        return
    z = np.abs(actual[valid] - predicted[valid]) / std[valid]
    nominal = np.linspace(0.1, 0.99, 30)
    # For normal predictions, central interval coverage p corresponds to z quantile.
    from scipy.stats import norm

    z_thresholds = norm.ppf((nominal + 1.0) / 2.0)
    empirical = np.array([(z <= threshold).mean() for threshold in z_thresholds])

    plt.figure(figsize=(5, 5))
    plt.plot(nominal, empirical, marker="o", markersize=3)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("Nominal coverage")
    plt.ylabel("Empirical coverage")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    _savefig(path)


def plot_residuals_by_parameter(
    parameters: pd.DataFrame,
    residuals: np.ndarray,
    path: str | Path,
    *,
    objective_name: str,
) -> None:
    """Plot validation residuals against every parameter."""
    residuals = np.asarray(residuals, dtype=float)
    n_params = len(parameters.columns)
    if n_params == 0:
        return
    fig, axes = plt.subplots(1, n_params, figsize=(max(3 * n_params, 6), 3), squeeze=False)
    for ax, column in zip(axes[0], parameters.columns):
        ax.scatter(parameters[column], residuals, s=28, edgecolor="black", linewidth=0.3, alpha=0.75)
        ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel(column)
        ax.set_ylabel("Residual")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle(f"Residuals by Parameter: {objective_name}")
    _savefig(path)
