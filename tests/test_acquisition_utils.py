"""Tests for acquisition.utilities: objective-file ordering, the constrained
Latin hypercube design, and the experimental database bookkeeping.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from config import ConstraintConfig, ExperimentConfig
from acquisition.utils import (
    generate_constrained_lhd,
    update_experimental_database,
)
from models.gp_fitting import sort_objective_files
from data.transformation import build_transformer

N_PARAMS = len(ExperimentConfig.PARAMETER_NAMES)
LB = ExperimentConfig.PARAMETER_BOUNDS[:, 0]
UB = ExperimentConfig.PARAMETER_BOUNDS[:, 1]
S = ConstraintConfig.SOLUBILIZATION_UREA
DIL_IDX = ConstraintConfig.DILUTION_FACTOR_IDX
UREA_IDX = ConstraintConfig.FINAL_UREA_IDX


def test_sort_objective_files_orders_numerically():
    files = ["model_10_b.pth", "model_2_a.pth", "model_1_c.pth", "scaler_1_x.pkl"]
    assert sort_objective_files(files) == [
        "model_1_c.pth",
        "scaler_1_x.pkl",  # same index -> lexicographic tie-break
        "model_2_a.pth",
        "model_10_b.pth",
    ]
    # Names without an index sort after indexed ones, lexicographically
    assert sort_objective_files(["legacy.pth", "model_2_a.pth"]) == [
        "model_2_a.pth",
        "legacy.pth",
    ]


@pytest.fixture
def bounds_tensor():
    return torch.tensor(np.vstack([LB, UB]), dtype=torch.float64)


def test_constrained_lhd_feasible_and_within_bounds(bounds_tensor):
    transformer = build_transformer(ExperimentConfig)
    samples = generate_constrained_lhd(
        n_samples=12, bounds=bounds_tensor, transformer=transformer,
        seed=7, use_maximin=False,
    )

    assert samples.shape == (12, N_PARAMS)
    assert (samples >= LB - 1e-12).all()
    assert (samples <= UB + 1e-12).all()
    # Every design point satisfies final_urea * dilution_factor >= S
    values = samples[:, UREA_IDX] * samples[:, DIL_IDX] - S
    assert (values >= -1e-9).all()


def test_constrained_lhd_stratifies_dilution(bounds_tensor):
    """The dilution column remains a Latin hypercube (one sample per stratum)."""
    transformer = build_transformer(ExperimentConfig)
    samples = generate_constrained_lhd(
        n_samples=12, bounds=bounds_tensor, transformer=transformer,
        seed=7, use_maximin=False,
    )
    dilution_unit = np.asarray(
        transformer.physical_to_unit_user(samples[:, [DIL_IDX]], cols=[DIL_IDX])
    ).flatten()
    strata = np.floor(dilution_unit * 12).astype(int)
    assert sorted(strata.tolist()) == list(range(12))


def test_constrained_lhd_reproducible_with_same_seed(bounds_tensor):
    transformer = build_transformer(ExperimentConfig)
    kwargs = dict(n_samples=10, bounds=bounds_tensor, transformer=transformer,
                  seed=123, use_maximin=False)
    np.testing.assert_array_equal(generate_constrained_lhd(**kwargs),
                                  generate_constrained_lhd(**kwargs))


def test_constrained_lhd_maximin_keeps_feasibility(bounds_tensor):
    transformer = build_transformer(ExperimentConfig)
    samples = generate_constrained_lhd(
        n_samples=10, bounds=bounds_tensor, transformer=transformer,
        seed=5, n_candidates=5, use_maximin=True,
    )
    values = samples[:, UREA_IDX] * samples[:, DIL_IDX] - S
    assert (values >= -1e-9).all()
    assert (samples >= LB - 1e-12).all()
    assert (samples <= UB + 1e-12).all()


def test_update_experimental_database_accumulates(tmp_path):
    path = str(tmp_path / "database.xlsx")

    # Non-integral values so dtypes survive the Excel round-trip
    first = pd.DataFrame({"DTT [mM]": [1.25, 2.5]})
    out = update_experimental_database(first, 0, path)
    assert list(out.columns) == ["DTT [mM]", "Iteration"]
    assert out["Iteration"].tolist() == [0, 0]
    # The caller's DataFrame is not mutated
    assert "Iteration" not in first.columns

    second = pd.DataFrame({"DTT [mM]": [3.75]})
    out = update_experimental_database(second, 1, path)
    assert len(out) == 3
    assert out["Iteration"].tolist() == [0, 0, 1]

    # The file on disk matches the accumulated frame
    on_disk = pd.read_excel(path)
    pd.testing.assert_frame_equal(on_disk, out)
