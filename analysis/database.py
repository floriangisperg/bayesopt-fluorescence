"""Experiment database helpers for campaign state management."""

from __future__ import annotations

from typing import Sequence

import pandas as pd


def validate_experimental_dataframe(
    df: pd.DataFrame,
    parameter_names: Sequence[str],
    objective_names: Sequence[str],
    require_objectives: bool = False,
) -> None:
    """Validate required columns and numeric values."""
    required = set(parameter_names)
    if require_objectives:
        required |= set(objective_names)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for column in list(parameter_names) + list(objective_names):
        if column in df.columns:
            numeric = pd.to_numeric(df[column], errors="coerce")
            if require_objectives or column in parameter_names:
                invalid = numeric.isna() & df[column].notna()
                if invalid.any():
                    raise ValueError(f"Column {column!r} contains non-numeric values")


def add_experiment_ids(df: pd.DataFrame, prefix: str = "EXP") -> pd.DataFrame:
    """Return a copy with stable experiment IDs if absent."""
    result = df.copy()
    if "Experiment ID" not in result.columns:
        result.insert(0, "Experiment ID", [f"{prefix}-{i + 1:04d}" for i in range(len(result))])
    return result


def mark_duplicate_parameters(
    df: pd.DataFrame,
    parameter_names: Sequence[str],
    decimals: int = 6,
) -> pd.DataFrame:
    """Add duplicate flags based on rounded parameter values."""
    result = df.copy()
    rounded = result[list(parameter_names)].astype(float).round(decimals)
    result["Duplicate Parameters"] = rounded.duplicated(keep=False)
    return result
