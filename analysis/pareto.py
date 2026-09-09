"""Pareto-front and hypervolume utilities for campaign analysis."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def _direction_signs(objective_directions: Sequence[str], n_objectives: int) -> np.ndarray:
    if len(objective_directions) != n_objectives:
        raise ValueError("Number of objective directions must match objective columns")
    signs = []
    for direction in objective_directions:
        normalized = direction.lower()
        if normalized == "maximize":
            signs.append(1.0)
        elif normalized == "minimize":
            signs.append(-1.0)
        else:
            raise ValueError(f"Unsupported objective direction: {direction!r}")
    return np.asarray(signs, dtype=float)


def to_maximization(values: np.ndarray, objective_directions: Sequence[str]) -> np.ndarray:
    """Convert objective values so larger is better for all objectives."""
    values = np.asarray(values, dtype=float)
    return values * _direction_signs(objective_directions, values.shape[1])


def is_non_dominated(values: np.ndarray, objective_directions: Sequence[str] | None = None) -> np.ndarray:
    """Return a boolean mask for non-dominated rows.

    All objectives are treated as maximization unless directions are provided.
    Rows containing NaNs are marked dominated.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must be a 2D array")
    if objective_directions is None:
        objective_directions = ["maximize"] * values.shape[1]

    finite_mask = np.isfinite(values).all(axis=1)
    scores = to_maximization(values[finite_mask], objective_directions)
    non_dominated = np.ones(scores.shape[0], dtype=bool)

    for i, point in enumerate(scores):
        if not non_dominated[i]:
            continue
        dominates_i = np.all(scores >= point, axis=1) & np.any(scores > point, axis=1)
        dominates_i[i] = False
        if np.any(dominates_i):
            non_dominated[i] = False

    result = np.zeros(values.shape[0], dtype=bool)
    result[np.where(finite_mask)[0]] = non_dominated
    return result


def auto_reference_point(
    values: np.ndarray,
    objective_directions: Sequence[str] | None = None,
    margin_fraction: float = 0.1,
) -> np.ndarray:
    """Choose a conservative reference point dominated by observed data."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values).all(axis=1)]
    if values.size == 0:
        raise ValueError("Cannot infer a reference point without finite objective values")
    if objective_directions is None:
        objective_directions = ["maximize"] * values.shape[1]
    scores = to_maximization(values, objective_directions)
    span = np.ptp(scores, axis=0)
    margin = np.where(span > 0, span * margin_fraction, 1.0)
    return scores.min(axis=0) - margin


def compute_hypervolume_2d(
    values: np.ndarray,
    reference_point: Sequence[float],
    objective_directions: Sequence[str] | None = None,
) -> float:
    """Compute dominated hypervolume for two objectives.

    The returned value is in maximization-oriented objective space. For minimization
    objectives, values are sign-flipped before hypervolume calculation.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Only two-objective hypervolume is supported")
    if objective_directions is None:
        objective_directions = ["maximize", "maximize"]
    scores = to_maximization(values, objective_directions)
    scores = scores[np.isfinite(scores).all(axis=1)]
    if scores.size == 0:
        return 0.0

    ref = np.asarray(reference_point, dtype=float)
    if ref.shape != (2,):
        raise ValueError("reference_point must contain two values")

    scores = scores[(scores[:, 0] > ref[0]) & (scores[:, 1] > ref[1])]
    if scores.size == 0:
        return 0.0

    scores = scores[is_non_dominated(scores, ["maximize", "maximize"])]
    scores = scores[np.argsort(scores[:, 0])[::-1]]

    hv = 0.0
    previous_y = ref[1]
    for x, y in scores:
        if y > previous_y:
            hv += max(x - ref[0], 0.0) * (y - previous_y)
            previous_y = y
    return float(hv)


def summarize_campaign_progress(
    df: pd.DataFrame,
    objective_names: Sequence[str],
    objective_directions: Sequence[str] | None,
    reference_point: Sequence[float],
    iteration_column: str = "Iteration",
) -> pd.DataFrame:
    """Summarize best values, Pareto count, and hypervolume by iteration."""
    if objective_directions is None:
        objective_directions = ["maximize"] * len(objective_names)

    if iteration_column in df.columns:
        iterations: Iterable[int] = sorted(df[iteration_column].dropna().astype(int).unique())
    else:
        iterations = [0]

    records = []
    for iteration in iterations:
        subset = df if iteration_column not in df.columns else df[df[iteration_column] <= iteration]
        values = subset[list(objective_names)].to_numpy(dtype=float)
        finite_values = values[np.isfinite(values).all(axis=1)]
        if finite_values.size == 0:
            continue

        record = {
            "Iteration": int(iteration),
            "Experiments": int(len(subset)),
            "Completed Experiments": int(len(finite_values)),
            "Pareto Points": int(is_non_dominated(finite_values, objective_directions).sum()),
            "Hypervolume": compute_hypervolume_2d(finite_values, reference_point, objective_directions),
        }
        signs = _direction_signs(objective_directions, len(objective_names))
        for i, objective_name in enumerate(objective_names):
            oriented = finite_values[:, i] * signs[i]
            best_idx = int(np.argmax(oriented))
            record[f"Best {objective_name}"] = float(finite_values[best_idx, i])
        records.append(record)

    return pd.DataFrame.from_records(records)
