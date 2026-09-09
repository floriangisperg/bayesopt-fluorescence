"""Report builders for candidates, validation, and campaign progress."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from .pareto import is_non_dominated


def _nearest_distances(unit_candidates: np.ndarray, unit_existing: np.ndarray) -> np.ndarray:
    if unit_existing.size == 0:
        return np.full(unit_candidates.shape[0], np.nan)
    distances = np.linalg.norm(unit_candidates[:, None, :] - unit_existing[None, :, :], axis=-1)
    return distances.min(axis=1)


def build_candidate_report(
    candidates_df: pd.DataFrame,
    parameter_names: Sequence[str],
    objective_names: Sequence[str],
    *,
    predicted_means: np.ndarray | None = None,
    predicted_stds: np.ndarray | None = None,
    acquisition_values: np.ndarray | None = None,
    existing_unit_x: torch.Tensor | None = None,
    candidate_unit_x: torch.Tensor | None = None,
    objective_directions: Sequence[str] | None = None,
    constraint_margins: np.ndarray | None = None,
    urea_refolding: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build a candidate-level report suitable for Excel export."""
    report = candidates_df[list(parameter_names)].copy()
    report.insert(0, "Candidate", np.arange(1, len(report) + 1))

    if predicted_means is not None:
        predicted_means = np.asarray(predicted_means, dtype=float)
        for i, objective_name in enumerate(objective_names):
            report[f"Predicted Mean {objective_name}"] = predicted_means[:, i]

        if objective_directions is None:
            objective_directions = ["maximize"] * len(objective_names)
        report["Non-dominated Predicted Candidate"] = is_non_dominated(
            predicted_means, objective_directions
        )

    if predicted_stds is not None:
        predicted_stds = np.asarray(predicted_stds, dtype=float)
        for i, objective_name in enumerate(objective_names):
            report[f"Predicted Std {objective_name}"] = predicted_stds[:, i]

    if acquisition_values is not None:
        report["Acquisition Value"] = np.asarray(acquisition_values, dtype=float)

    if existing_unit_x is not None and candidate_unit_x is not None:
        report["Nearest Previous Distance (unit space)"] = _nearest_distances(
            candidate_unit_x.detach().cpu().numpy(),
            existing_unit_x.detach().cpu().numpy(),
        )

    if constraint_margins is not None:
        report["Urea Constraint Margin"] = np.asarray(constraint_margins, dtype=float)

    if urea_refolding is not None:
        report["Urea Refolding [M]"] = np.asarray(urea_refolding, dtype=float)

    return report


def save_dataframe_report(df: pd.DataFrame, path: str | Path) -> None:
    """Save a DataFrame as Excel or CSV based on suffix."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_excel(path, index=False)


def save_json_report(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=str)
