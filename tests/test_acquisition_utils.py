"""Tests for acquisition.utilities: objective-file ordering, the constrained
Latin hypercube design, and the experimental database bookkeeping.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from acquisition.utils import (
    generate_constrained_lhd,
    generate_initial_design,
    update_experimental_database,
)
from config import ConstraintConfig, ExperimentConfig
from constraints.urea_dilution import urea_constraint_callable
from data.transformation import build_transformer
from models.gp_fitting import sort_objective_files

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


def test_constrained_lhd_restricts_dilution_when_partially_infeasible(bounds_tensor):
    """With S=13 M, dilution < 13/6 has no feasible urea: draws must stay in
    the feasible dilution subrange instead of producing out-of-bounds urea."""
    transformer = build_transformer(ExperimentConfig)
    samples = generate_constrained_lhd(
        n_samples=15, bounds=bounds_tensor, transformer=transformer,
        seed=11, use_maximin=False, solubilization_urea=13.0,
    )
    assert (samples >= LB - 1e-12).all()
    assert (samples <= UB + 1e-12).all()
    values = samples[:, UREA_IDX] * samples[:, DIL_IDX] - 13.0
    assert (values >= -1e-9).all()
    # Every dilution factor comes from the feasible subrange [13/6, 40]
    assert (samples[:, DIL_IDX] >= 13.0 / 6.0 - 1e-9).all()


def test_constrained_lhd_raises_when_config_infeasible(bounds_tensor):
    """S=300 M exceeds even urea_upper * dilution_max = 240: no design exists."""
    transformer = build_transformer(ExperimentConfig)
    with pytest.raises(ValueError, match="cannot be satisfied"):
        generate_constrained_lhd(
            n_samples=8, bounds=bounds_tensor, transformer=transformer,
            seed=1, use_maximin=False, solubilization_urea=300.0,
        )


def test_design_strategies_dispatch_explicitly(bounds_tensor):
    transformer = build_transformer(ExperimentConfig)

    # constrained_lhd works without any callable (the strategy is explicit)
    samples = generate_initial_design(
        n_samples=6, bounds=bounds_tensor, transformer=transformer,
        seed=3, n_candidates=3, use_maximin=False, design_strategy="constrained_lhd",
    )
    values = samples[:, UREA_IDX] * samples[:, DIL_IDX] - S
    assert (values >= -1e-9).all()

    # plain LHS ignores constraints
    samples = generate_initial_design(
        n_samples=6, bounds=bounds_tensor, transformer=transformer,
        seed=3, design_strategy="lhs",
    )
    assert samples.shape == (6, N_PARAMS)

    # a callable supplied alongside a strategy that would ignore it is an error
    with pytest.raises(ValueError, match="would ignore it"):
        generate_initial_design(
            n_samples=6, bounds=bounds_tensor, transformer=transformer,
            seed=3, design_strategy="lhs",
            constraint_callable=urea_constraint_callable,
        )

    # rejection requires a callable; unknown strategies are rejected
    with pytest.raises(ValueError, match="requires a constraint_callable"):
        generate_initial_design(
            n_samples=6, bounds=bounds_tensor, transformer=transformer,
            seed=3, design_strategy="rejection",
        )
    with pytest.raises(ValueError, match="Unknown design_strategy"):
        generate_initial_design(
            n_samples=6, bounds=bounds_tensor, transformer=transformer,
            seed=3, design_strategy="name-sniffing",
        )


def test_rejection_strategy_is_fully_seeded(bounds_tensor):
    """The maximin subset draw must be reproducible from `seed` alone."""
    transformer = build_transformer(ExperimentConfig)
    kwargs = dict(
        n_samples=6, bounds=bounds_tensor, transformer=transformer,
        seed=17, n_candidates=5, use_maximin=True,
        design_strategy="rejection",
        constraint_callable=urea_constraint_callable,
    )
    first = generate_initial_design(**kwargs)
    second = generate_initial_design(**kwargs)
    torch.testing.assert_close(first, second)

    values = urea_constraint_callable(first)
    assert (values > 0).all()


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


@pytest.mark.parametrize("strategy", ["lhs", "constrained_lhd", "rejection"])
def test_single_sample_design(bounds_tensor, strategy):
    samples = generate_initial_design(
        1, bounds_tensor, build_transformer(ExperimentConfig), design_strategy=strategy,
        constraint_callable=urea_constraint_callable if strategy == "rejection" else None,
    )
    assert samples.shape == (1, N_PARAMS)
    assert torch.isfinite(samples).all()


@pytest.mark.parametrize("name", ["n_samples", "n_candidates", "oversampling_factor"])
@pytest.mark.parametrize("value", [0, -1, 1.5])
def test_invalid_design_counts(bounds_tensor, name, value):
    kwargs = {"n_samples": 4, "n_candidates": 2, "oversampling_factor": 2}
    kwargs[name] = value
    with pytest.raises(ValueError, match=name):
        generate_initial_design(bounds=bounds_tensor, transformer=build_transformer(ExperimentConfig), **kwargs)


def test_rejection_stops_after_enough_points_and_accepts_boundary(bounds_tensor):
    calls = []

    def feasible(samples):
        calls.append(len(samples))
        return torch.zeros(len(samples))

    result = generate_initial_design(
        4, bounds_tensor, build_transformer(ExperimentConfig), design_strategy="rejection",
        constraint_callable=feasible, oversampling_factor=2,
    )
    assert result.shape == (4, N_PARAMS)
    assert len(calls) == 1


@pytest.mark.parametrize("accepted", [0, 1])
def test_rejection_raises_instead_of_returning_short_design(bounds_tensor, accepted):
    calls = 0

    def scarce(samples):
        nonlocal calls
        result = torch.full((len(samples),), -1.0)
        if calls == 0:
            result[:accepted] = 1.0
        calls += 1
        return result

    with pytest.raises(RuntimeError, match=f"only {accepted} feasible samples"):
        generate_initial_design(
            4, bounds_tensor, build_transformer(ExperimentConfig), design_strategy="rejection",
            constraint_callable=scarce, oversampling_factor=1,
        )
