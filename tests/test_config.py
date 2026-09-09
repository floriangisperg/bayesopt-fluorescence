"""Tests for config.py helpers."""

import numpy as np
import torch

from config import (
    ExperimentConfig,
    OptimizationConfig,
    get_normalized_bounds,
    get_optimization_params,
    get_transposed_bounds,
)


def test_get_transposed_bounds_matches_config():
    bounds = get_transposed_bounds()
    assert bounds.shape == (2, len(ExperimentConfig.PARAMETER_NAMES))
    assert bounds.dtype == torch.float64
    np.testing.assert_array_equal(bounds.numpy()[0],
                                  ExperimentConfig.PARAMETER_BOUNDS[:, 0])
    np.testing.assert_array_equal(bounds.numpy()[1],
                                  ExperimentConfig.PARAMETER_BOUNDS[:, 1])


def test_get_normalized_bounds():
    bounds = get_normalized_bounds(5)
    assert bounds.shape == (2, 5)
    np.testing.assert_array_equal(bounds[0], np.zeros(5))
    np.testing.assert_array_equal(bounds[1], np.ones(5))


def test_get_optimization_params_defaults(monkeypatch):
    monkeypatch.delenv("SMOKE_TEST", raising=False)
    params = get_optimization_params()
    assert params["mc_samples"] == OptimizationConfig.MC_SAMPLES
    assert params["num_restarts"] == OptimizationConfig.NUM_RESTARTS
    assert params["raw_samples"] == OptimizationConfig.RAW_SAMPLES


def test_get_optimization_params_smoke_mode(monkeypatch):
    monkeypatch.setenv("SMOKE_TEST", "1")
    assert get_optimization_params() == {
        "mc_samples": 500,
        "num_restarts": 3,
        "raw_samples": 24,
    }
