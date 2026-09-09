"""Tests for data.transformation: scaler math and ParameterTransformer wiring.

These verify the invariants the whole pipeline relies on: every configured
transformation maps the physical bounds onto the unit interval and round-trips
exactly, for NumPy arrays and torch tensors alike.
"""

import numpy as np
import pytest
import torch

from config import ExperimentConfig
from data.transformation import (
    TRANSFORM_REGISTRY,
    IdentityTransform,
    LinearScaler,
    LogScaler,
    ReciprocalScaler,
    ParameterTransformer,
    build_transformer,
)

N_PARAMS = len(ExperimentConfig.PARAMETER_NAMES)

# (registry kind, physical bounds) pairs covering every configured transform
# plus the unconfigured ones (reciprocal alias, log, logit).
SCALER_CASES = [
    ("linear", (2.0, 40.0)),
    ("1/x", (2.0, 40.0)),
    ("reciprocal", (2.0, 40.0)),
    ("log", (0.1, 25.0)),
    ("logit", (0.0, 2.5)),
]


def _random_physical(n, seed):
    rng = np.random.default_rng(seed)
    lb = ExperimentConfig.PARAMETER_BOUNDS[:, 0]
    ub = ExperimentConfig.PARAMETER_BOUNDS[:, 1]
    return rng.uniform(lb, ub, size=(n, N_PARAMS))


@pytest.mark.parametrize("kind,bounds", SCALER_CASES)
def test_scaler_round_trip(kind, bounds):
    """physical -> unit -> physical reproduces the input exactly."""
    transform = TRANSFORM_REGISTRY[kind](*bounds)
    rng = np.random.default_rng(0)
    x = rng.uniform(bounds[0], bounds[1], size=200)

    for values in (x, torch.from_numpy(x)):
        back = transform.unit_to_physical(transform.physical_to_unit(values))
        if torch.is_tensor(back):
            back = back.numpy()
        np.testing.assert_allclose(back, x, atol=1e-10)


@pytest.mark.parametrize("kind,bounds", SCALER_CASES)
def test_scaler_maps_bounds_onto_unit_interval(kind, bounds):
    transform = TRANSFORM_REGISTRY[kind](*bounds)
    z = transform.physical_to_unit(np.array(bounds))
    z = z.numpy() if torch.is_tensor(z) else z
    assert z[0] == pytest.approx(0.0, abs=1e-12)
    assert z[1] == pytest.approx(1.0, abs=1e-12)
    assert transform.transformed_bounds() == (pytest.approx(0.0, abs=1e-12),
                                              pytest.approx(1.0, abs=1e-12))


def test_identity_transform_passthrough():
    transform = IdentityTransform(2.0, 40.0)
    x = np.array([3.0, 17.5])
    np.testing.assert_array_equal(transform.physical_to_unit(x), x)
    np.testing.assert_array_equal(transform.unit_to_physical(x), x)


def test_scalers_reject_invalid_bounds():
    with pytest.raises(ValueError):
        LinearScaler(5.0, 5.0)  # lower == upper
    with pytest.raises(ValueError):
        LinearScaler(np.inf, 5.0)  # non-finite
    with pytest.raises(ValueError):
        ReciprocalScaler(0.0, 5.0)  # zero bound
    with pytest.raises(ValueError):
        ReciprocalScaler(-1.0, 1.0)  # bounds cross zero
    with pytest.raises(ValueError):
        LogScaler(0.0, 5.0)  # non-positive lower bound


def test_experiment_config_transformer_round_trip_both_spaces():
    transformer = build_transformer(ExperimentConfig)
    X = _random_physical(50, seed=1)

    for forward, inverse in (
        (transformer.physical_to_unit_model, transformer.unit_to_physical_model),
        (transformer.physical_to_unit_user, transformer.unit_to_physical_user),
    ):
        back = inverse(forward(X, as_tensor=True), as_tensor=True)
        assert torch.is_tensor(back)
        np.testing.assert_allclose(back.numpy(), X, atol=1e-10)

        back_np = inverse(forward(X))
        assert isinstance(back_np, np.ndarray)
        np.testing.assert_allclose(back_np, X, atol=1e-10)


def test_experiment_config_spaces_span_unit_interval():
    transformer = build_transformer(ExperimentConfig)
    for getter in (transformer.get_model_bounds, transformer.get_user_bounds):
        bounds = getter()
        assert bounds.shape == (2, N_PARAMS)
        np.testing.assert_allclose(bounds[0], 0.0, atol=1e-12)
        np.testing.assert_allclose(bounds[1], 1.0, atol=1e-12)
    np.testing.assert_allclose(
        transformer.get_physical_bounds(),
        ExperimentConfig.PARAMETER_BOUNDS.T,
    )


def test_column_subset_matches_full_transform():
    transformer = build_transformer(ExperimentConfig)
    X = _random_physical(20, seed=2)

    for forward in (transformer.physical_to_unit_model,
                    transformer.physical_to_unit_user):
        full = forward(X)
        for j in range(N_PARAMS):
            sub = forward(X[:, j], cols=[j])
            np.testing.assert_allclose(np.asarray(sub).flatten(), full[:, j],
                                       atol=1e-12)


def test_transformer_shape_rules():
    transformer = build_transformer(ExperimentConfig)
    row = _random_physical(1, seed=3)[0]

    # 1D input with all columns is a single row
    out = transformer.physical_to_unit_model(row)
    assert out.shape == (N_PARAMS,)

    # 1D input with one selected column is many values for that parameter
    values = transformer.physical_to_unit_model(
        np.full(7, row[0]), cols=[0])
    assert np.asarray(values).shape == (7,)

    # A scalar with one selected column stays a scalar
    scalar = transformer.physical_to_unit_model(float(row[0]), cols=[0])
    assert np.asarray(scalar).ndim == 0

    # 1D input whose length matches neither is an error
    with pytest.raises(ValueError):
        transformer.physical_to_unit_model(np.ones(3), cols=[0, 1, 2, 3])

    with pytest.raises(IndexError):
        transformer.physical_to_unit_model(row, cols=[N_PARAMS])
    with pytest.raises(ValueError):
        transformer.physical_to_unit_model(row, cols=[])


def test_transformer_rejects_inconsistent_config():
    transforms = {
        "a": {"user_space": "linear", "model_space": "linear"},
        "b": {"user_space": "linear", "model_space": "linear"},
    }
    bounds = np.array([[0.0, 1.0], [2.0, 3.0]])

    # transforms missing one parameter
    with pytest.raises(ValueError):
        ParameterTransformer(["a", "b"], bounds,
                             {"a": transforms["a"]})
    # bounds with the wrong shape
    with pytest.raises(ValueError):
        ParameterTransformer(["a", "b"], bounds[:1], transforms)
    # unknown transform kind
    bad = dict(transforms, a={"user_space": "linear", "model_space": "quadratic"})
    with pytest.raises(ValueError):
        ParameterTransformer(["a", "b"], bounds, bad)
