"""Experiment database helpers for campaign state management."""

from __future__ import annotations

from typing import Sequence
from uuid import uuid4

import numpy as np
import pandas as pd


def validate_experimental_dataframe(
    df: pd.DataFrame,
    parameter_names: Sequence[str],
    objective_names: Sequence[str],
    require_objectives: bool = False,
) -> None:
    """Validate required columns and numeric values."""
    if df.empty:
        raise ValueError("Experimental data must contain at least one row")
    required = set(parameter_names)
    if require_objectives:
        required |= set(objective_names)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for column in list(parameter_names) + list(objective_names):
        if column in df.columns:
            numeric = pd.to_numeric(df[column], errors="coerce")
            invalid = ~np.isfinite(numeric)
            if column not in parameter_names:
                invalid &= df[column].notna()  # Unmeasured objectives are allowed here.
            if invalid.any():
                raise ValueError(
                    f"Column {column!r} contains missing, non-numeric or non-finite values "
                    f"in rows {df.index[invalid].tolist()}"
                )

    # Apply physical checks when these are the configured experimental inputs.
    from config import ConstraintConfig, ExperimentConfig
    from constraints import assert_urea_feasible

    if list(parameter_names) == list(ExperimentConfig.PARAMETER_NAMES):
        values = df[list(parameter_names)].to_numpy(dtype=float)
        bounds = ExperimentConfig.PARAMETER_BOUNDS
        # Allow only round-off from candidate transforms and Excel export.
        outside = ((values < bounds[:, 0] - 1e-9) | (values > bounds[:, 1] + 1e-9)).any(axis=1)
        if outside.any():
            raise ValueError(f"Parameters outside configured bounds in rows {df.index[outside].tolist()}")
        if ConstraintConfig.ENABLE_UREA_CONSTRAINT:
            assert_urea_feasible(values)


def add_experiment_ids(df: pd.DataFrame, prefix: str = "EXP") -> pd.DataFrame:
    """Assign globally unique IDs to new rows, preserving existing IDs."""
    result = df.copy()
    if "Experiment ID" not in result.columns:
        result.insert(0, "Experiment ID", [f"{prefix}-{uuid4().hex}" for _ in range(len(result))])
    else:
        missing = result["Experiment ID"].isna() | result["Experiment ID"].eq("")
        result.loc[missing, "Experiment ID"] = [f"{prefix}-{uuid4().hex}" for _ in range(missing.sum())]
    if result["Experiment ID"].duplicated().any():
        raise ValueError("Duplicate Experiment ID values")
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
