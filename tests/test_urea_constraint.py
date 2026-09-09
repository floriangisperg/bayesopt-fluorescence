"""Tests for the urea dilution constraint.

The core invariant: the linear half-space returned by
``get_urea_linear_constraint`` (expressed in unit model space, where the
dilution factor is stored reciprocally) must be *exactly* equivalent to the
physical feasibility condition ``final_urea * dilution_factor >= S`` —
including on the boundary, which the constraint treats as feasible.
"""

import itertools

import numpy as np
import pytest
import torch

from config import ConstraintConfig, ExperimentConfig
from constraints.urea_dilution import (
    CONSTRAINT_TOLERANCE,
    assert_urea_feasible,
    calculate_urea_refolding_concentration,
    get_urea_linear_constraint,
    urea_constraint_callable,
)
from data.transformation import build_transformer

N_PARAMS = len(ExperimentConfig.PARAMETER_NAMES)
LB = ExperimentConfig.PARAMETER_BOUNDS[:, 0]
UB = ExperimentConfig.PARAMETER_BOUNDS[:, 1]
S = ConstraintConfig.SOLUBILIZATION_UREA
DIL_IDX = ConstraintConfig.DILUTION_FACTOR_IDX
UREA_IDX = ConstraintConfig.FINAL_UREA_IDX


def _random_physical(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(LB, UB, size=(n, N_PARAMS))


def _linear_values(points):
    """Evaluate the acquisition-space linear form at physical points."""
    transformer = build_transformer(ExperimentConfig)
    indices, coefficients, rhs = get_urea_linear_constraint()
    unit = transformer.physical_to_unit_model(points, as_tensor=True).double()
    return (unit[:, indices] @ coefficients.double() - rhs).numpy()


def _physical_values(points):
    return points[:, UREA_IDX] * points[:, DIL_IDX] - S


def test_linear_constraint_equivalent_to_physical_condition():
    """Sign agreement away from the boundary, over a dense sample of the box."""
    points = _random_physical(5000, seed=42)

    # corners of the parameter box
    corners = np.array(list(itertools.product(*zip(LB, UB))))
    points = np.vstack([points, corners])

    physical = _physical_values(points)
    linear = _linear_values(points)

    off_boundary = np.abs(physical) > 1e-7
    assert off_boundary.sum() > 1000  # the sample genuinely straddles the boundary
    np.testing.assert_array_equal(np.sign(linear[off_boundary]),
                                  np.sign(physical[off_boundary]))


def test_linear_constraint_boundary_is_feasible():
    """Points exactly on final_urea = S / dilution satisfy the linear form.

    The boundary corresponds to a zero-urea refolding buffer, which is
    preparable and therefore must count as feasible.
    """
    rng = np.random.default_rng(7)
    points = _random_physical(500, seed=8)
    dilution = rng.uniform(LB[DIL_IDX], UB[DIL_IDX], size=len(points))
    points[:, DIL_IDX] = dilution
    points[:, UREA_IDX] = S / dilution  # exactly on the boundary

    np.testing.assert_allclose(_physical_values(points), 0.0, atol=1e-12)
    linear = _linear_values(points)
    assert (linear > -1e-9).all()


def test_linear_constraint_just_off_the_boundary():
    """Eps-feasible/infeasible perturbations keep their sign in both forms."""
    rng = np.random.default_rng(9)
    points = _random_physical(200, seed=10)
    dilution = rng.uniform(LB[DIL_IDX] + 1.0, UB[DIL_IDX], size=len(points))
    points[:, DIL_IDX] = dilution

    for factor, expected_sign in ((1.0 + 1e-6, +1), (1.0 - 1e-6, -1)):
        perturbed = points.copy()
        perturbed[:, UREA_IDX] = factor * S / dilution
        physical = _physical_values(perturbed)
        linear = _linear_values(perturbed)
        assert (np.sign(physical) == expected_sign).all()
        np.testing.assert_array_equal(np.sign(linear), np.sign(physical))


def test_linear_constraint_rejects_config_drift(monkeypatch):
    """The linear form only holds under reciprocal dilution / linear urea.

    Renaming either transform in the config must fail loudly instead of
    silently building a wrong constraint.
    """
    monkeypatch.setitem(ExperimentConfig.PARAMETER_TRANSFORMATION, "Dilution Factor",
                        {"user_space": "1/x", "model_space": "linear"})
    with pytest.raises(ValueError, match="model-space transform"):
        get_urea_linear_constraint()

    monkeypatch.setitem(ExperimentConfig.PARAMETER_TRANSFORMATION, "Dilution Factor",
                        {"user_space": "1/x", "model_space": "1/x"})
    monkeypatch.setitem(ExperimentConfig.PARAMETER_TRANSFORMATION, "Final Urea [M]",
                        {"user_space": "linear", "model_space": "1/x"})
    with pytest.raises(ValueError, match="model-space transform"):
        get_urea_linear_constraint()


def test_assert_urea_feasible_passes_and_returns_values():
    points = _random_physical(200, seed=11)
    # Force feasibility: enough urea for every dilution factor
    points[:, UREA_IDX] = np.maximum(points[:, UREA_IDX], S / points[:, DIL_IDX])

    values = assert_urea_feasible(points)
    np.testing.assert_allclose(values, _physical_values(points))
    assert (values >= -CONSTRAINT_TOLERANCE).all()

    # torch input and single 1D sample take the same code paths
    values_t = assert_urea_feasible(torch.from_numpy(points))
    np.testing.assert_allclose(values_t, values)
    single = assert_urea_feasible(points[0])
    assert single.shape == (1,)


def test_assert_urea_feasible_raises_with_details():
    points = _random_physical(10, seed=12)
    points[:, UREA_IDX] = 0.0  # 0 urea is always infeasible (0 * d < 8)

    with pytest.raises(ValueError, match="sample 0"):
        assert_urea_feasible(points)

    # All ten samples violate, and the error reports the count
    with pytest.raises(ValueError, match=r"10 of 10 samples"):
        assert_urea_feasible(points)


def test_assert_urea_feasible_tolerance_window():
    # Margins chosen away from exact float boundaries of the tolerance
    # Violation of 5e-7 * dilution units: inside the default tolerance -> passes
    within_tol = np.array([[0.0, 0.0, 2.0, 8.0, (S - 5e-7) / 2.0]])
    assert_urea_feasible(within_tol)

    # Violation of 1e-4 * dilution units: beyond the tolerance -> raises
    beyond_tol = np.array([[0.0, 0.0, 2.0, 8.0, (S - 1e-4) / 2.0]])
    with pytest.raises(ValueError):
        assert_urea_feasible(beyond_tol)

    # Explicit tolerance overrides the default
    assert_urea_feasible(beyond_tol, tolerance=1e-3)


def test_calculate_urea_refolding_concentration():
    # (final_urea * dilution - S) / (dilution - 1)
    assert calculate_urea_refolding_concentration(2.0, 4.0, 8.0) == pytest.approx(0.0)
    assert calculate_urea_refolding_concentration(3.0, 4.0, 8.0) == pytest.approx(4.0 / 3.0)
    # Default solubilization urea comes from the config
    assert calculate_urea_refolding_concentration(2.0, 2.0) == pytest.approx(
        (2.0 * 2.0 - S) / 1.0
    )
    with pytest.raises(ValueError):
        calculate_urea_refolding_concentration(1.0, 1.0)


def test_urea_constraint_callable_matches_physical_condition():
    points = _random_physical(50, seed=13)
    values = urea_constraint_callable(torch.from_numpy(points))
    expected = torch.from_numpy(_physical_values(points))
    torch.testing.assert_close(values, expected)

    # single sample returns a scalar tensor
    single = urea_constraint_callable(torch.from_numpy(points[0]))
    assert single.ndim == 0
    torch.testing.assert_close(single, expected[0])


def test_urea_constraint_callable_denormalizes_with_bounds():
    points = _random_physical(20, seed=14)
    unit = torch.from_numpy(
        (points - LB) / (UB - LB)
    )
    bounds = torch.tensor(np.vstack([LB, UB]), dtype=torch.float64)

    with_bounds = urea_constraint_callable(unit, bounds=bounds)
    direct = urea_constraint_callable(torch.from_numpy(points))
    torch.testing.assert_close(with_bounds, direct)
