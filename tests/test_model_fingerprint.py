"""Tests for checkpoint fingerprinting in models.gp_fitting.

A saved model is reconstructed with whatever data is passed at load time, so
the stored fingerprint of training data and configuration is the only guard
against pairing stale hyperparameters with a different problem.
"""

import gpytorch
import numpy as np
import pytest
import torch

from config import ExperimentConfig, ModelConfig
from data.preprocessing import save_scalers, standardize_objectives
from models import GPModel, fit_gp_model, load_gp_model, save_gp_model
from run_optimization import load_trained_models


@pytest.fixture(scope="module")
def tiny_training_data():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(8, len(ExperimentConfig.PARAMETER_NAMES)))
    y = rng.uniform(-1, 1, size=8)
    return torch.from_numpy(X).double(), torch.from_numpy(y).double()


@pytest.fixture(scope="module")
def fitted_model(tiny_training_data):
    train_x, train_y = tiny_training_data
    model, likelihood = fit_gp_model(
        train_x=train_x, train_y=train_y, model_class=GPModel,
        noise=ModelConfig.INITIAL_NOISE_LEVEL,
    )
    return model, likelihood


@pytest.fixture
def checkpoint(tmp_path, fitted_model):
    model, likelihood = fitted_model
    path = str(tmp_path / "model.pth")
    save_gp_model(model, likelihood, path)
    return path


def test_load_matches_same_data(checkpoint, tiny_training_data):
    train_x, train_y = tiny_training_data
    model, _ = load_gp_model(checkpoint, GPModel, train_x, train_y.reshape(-1, 1), 0)
    assert not model.training


def test_load_rejects_changed_targets(checkpoint, tiny_training_data):
    train_x, train_y = tiny_training_data
    tampered_y = train_y.clone()
    tampered_y[0] += 1.0
    with pytest.raises(ValueError, match="training targets differ"):
        load_gp_model(checkpoint, GPModel, train_x, tampered_y.reshape(-1, 1), 0)


def test_load_rejects_changed_inputs(checkpoint, tiny_training_data):
    train_x, train_y = tiny_training_data
    tampered_x = train_x.clone()
    tampered_x[0, 0] += 0.5
    with pytest.raises(ValueError, match="training inputs differ"):
        load_gp_model(checkpoint, GPModel, tampered_x, train_y.reshape(-1, 1), 0)


def test_load_rejects_changed_config(checkpoint, tiny_training_data, monkeypatch):
    train_x, train_y = tiny_training_data
    shifted_bounds = ExperimentConfig.PARAMETER_BOUNDS.copy()
    shifted_bounds[0, 0] = 1.0
    monkeypatch.setattr(ExperimentConfig, "PARAMETER_BOUNDS", shifted_bounds)
    with pytest.raises(ValueError, match="configuration differs"):
        load_gp_model(checkpoint, GPModel, train_x, train_y.reshape(-1, 1), 0)


def test_strict_false_downgrades_mismatch_to_warning(checkpoint, tiny_training_data, caplog):
    train_x, train_y = tiny_training_data
    tampered_y = train_y.clone()
    tampered_y[3] -= 2.0
    with caplog.at_level("WARNING", logger="models.gp_fitting"):
        model, _ = load_gp_model(
            checkpoint, GPModel, train_x, tampered_y.reshape(-1, 1), 0,
            strict=False,
        )
    assert not model.training
    assert any("training targets differ" in record.message for record in caplog.records)


def test_legacy_checkpoint_without_fingerprint_loads(tmp_path, fitted_model, tiny_training_data):
    """Checkpoints saved before fingerprinting still load (with a note)."""
    model, likelihood = fitted_model
    train_x, train_y = tiny_training_data

    legacy_path = str(tmp_path / "legacy.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "likelihood_state_dict": likelihood.state_dict(),
        },
        legacy_path,
    )
    loaded, _ = load_gp_model(legacy_path, GPModel, train_x, train_y.reshape(-1, 1), 0)
    assert not loaded.training


@pytest.mark.parametrize("scale,offset", [(1.0, 0.0), (1.0, 16.0), (2.0, 0.0)])
def test_loader_checks_raw_objective_scaling(tmp_path, scale, offset):
    train_x = torch.linspace(0, 1, 20, dtype=torch.float64).reshape(4, 5)
    raw_y = np.array([[0., 0.], [1., 2.], [2., 4.], [3., 6.]])
    original_y, original_scalers = standardize_objectives(raw_y)
    current_y, current_scalers = standardize_objectives(raw_y * scale + offset)
    # The old target hash cannot distinguish these measurement changes.
    assert torch.equal(original_y, current_y)
    for i in range(2):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GPModel(train_x, original_y[:, i], likelihood)
        save_gp_model(model, likelihood, str(tmp_path / f"model_{i+1}_test.pth"))
        save_scalers([original_scalers[i]], str(tmp_path / f"scaler_{i+1}_test.pkl"))
    if scale == 1 and offset == 0:
        model, _ = load_trained_models(str(tmp_path), train_x, current_y, current_scalers)
        assert len(model.models) == 2
    else:
        with pytest.raises(ValueError, match="scaler does not match"):
            load_trained_models(str(tmp_path), train_x, current_y, current_scalers)
