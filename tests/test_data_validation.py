"""Tests for the pre-flight data-quality check at the training boundary."""

import logging

import numpy as np
import pandas as pd
import pytest

from analysis.database import validate_experimental_dataframe
from config import ExperimentConfig
from data.preprocessing import validate_experiment_data


def _plan(n=4):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        rng.uniform(ExperimentConfig.PARAMETER_BOUNDS[:, 0],
                    ExperimentConfig.PARAMETER_BOUNDS[:, 1], size=(n, len(ExperimentConfig.PARAMETER_NAMES))),
        columns=ExperimentConfig.PARAMETER_NAMES,
    )
    df["Final Urea [M]"] = np.maximum(df["Final Urea [M]"], 8.0 / df["Dilution Factor"])
    df[ExperimentConfig.OBJECTIVE_NAMES[0]] = np.arange(n, dtype=float)
    df[ExperimentConfig.OBJECTIVE_NAMES[1]] = np.arange(n, dtype=float) + 1
    return df


def test_clean_plan_passes_silently():
    validate_experiment_data(_plan())


def test_missing_objective_raises_with_row_numbers():
    df = _plan()
    df.loc[2, ExperimentConfig.OBJECTIVE_NAMES[1]] = np.nan
    with pytest.raises(ValueError, match=r"rows \[2\]"):
        validate_experiment_data(df)


def test_multiple_missing_rows_all_listed():
    df = _plan(n=5)
    df.loc[0, ExperimentConfig.OBJECTIVE_NAMES[0]] = np.nan
    df.loc[3, ExperimentConfig.OBJECTIVE_NAMES[1]] = np.nan
    with pytest.raises(ValueError, match=r"rows \[0, 3\]"):
        validate_experiment_data(df)


def test_duplicate_parameters_warn(caplog):
    df = _plan(n=4)
    df.loc[3, ExperimentConfig.PARAMETER_NAMES] = df.loc[1, ExperimentConfig.PARAMETER_NAMES]
    with caplog.at_level(logging.WARNING, logger="data.preprocessing"):
        validate_experiment_data(df)
    assert any("Duplicate parameter rows" in record.message for record in caplog.records)
    # Duplicates alone do not block training (replicates can be intentional)
    assert not any(record.levelname == "ERROR" for record in caplog.records)


def test_distinct_objectives_same_parameters_also_flagged(caplog):
    """Identical parameters with different outcomes are still duplicates."""
    df = _plan(n=2)
    df.loc[1, ExperimentConfig.PARAMETER_NAMES] = df.loc[0, ExperimentConfig.PARAMETER_NAMES]
    with caplog.at_level(logging.WARNING, logger="data.preprocessing"):
        validate_experiment_data(df)
    assert any("Duplicate parameter rows" in record.message for record in caplog.records)


def test_custom_column_names():
    """Workshop notebooks define their own parameters/objectives; validation
    must check those columns instead of the config defaults."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.uniform(0, 1, size=(3, 2)), columns=["salt", "temp"])
    df["signal"] = [1.0, 2.0, 3.0]

    # Defaults fail (no config columns present); explicit names work
    with pytest.raises(KeyError):
        validate_experiment_data(df)
    validate_experiment_data(df, parameter_names=["salt", "temp"],
                             objective_names=["signal"])

    # ...and the custom names drive the missing-value check
    df.loc[2, "signal"] = np.nan
    with pytest.raises(ValueError, match=r"rows \[2\]"):
        validate_experiment_data(df, parameter_names=["salt", "temp"],
                                 objective_names=["signal"])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, -1.0, 26.0, "invalid"])
def test_invalid_parameters_rejected_by_both_validators(value):
    df = _plan().astype(object)
    df.loc[1, "DTT [mM]"] = value
    for validate in (
        validate_experiment_data,
        lambda frame: validate_experimental_dataframe(
            frame, ExperimentConfig.PARAMETER_NAMES, ExperimentConfig.OBJECTIVE_NAMES, True
        ),
    ):
        with pytest.raises(ValueError):
            validate(df)


def test_infeasible_training_data_rejected():
    df = _plan()
    df.loc[0, "Final Urea [M]"] = 0.0
    with pytest.raises(ValueError, match="Urea dilution constraint violated"):
        validate_experiment_data(df)


def test_disabled_constraint_allows_infeasible_training_data(monkeypatch):
    from config import ConstraintConfig

    monkeypatch.setattr(ConstraintConfig, "ENABLE_UREA_CONSTRAINT", False)
    df = _plan()
    df.loc[0, "Final Urea [M]"] = 0.0
    validate_experiment_data(df)


@pytest.mark.parametrize("value", [np.inf, -np.inf, "invalid"])
def test_invalid_objectives_rejected(value):
    df = _plan().astype(object)
    df.loc[0, ExperimentConfig.OBJECTIVE_NAMES[0]] = value
    with pytest.raises(ValueError):
        validate_experiment_data(df)


def test_suggested_rows_can_have_missing_objectives():
    df = _plan()
    df[ExperimentConfig.OBJECTIVE_NAMES] = np.nan
    validate_experimental_dataframe(
        df, ExperimentConfig.PARAMETER_NAMES, ExperimentConfig.OBJECTIVE_NAMES, True
    )


def test_export_roundoff_at_parameter_bound_is_allowed():
    df = _plan()
    df.loc[0, "DTT [mM]"] = -1e-12
    validate_experiment_data(df)


def test_empty_training_data_rejected():
    with pytest.raises(ValueError, match="at least one row"):
        validate_experiment_data(_plan().iloc[:0])
