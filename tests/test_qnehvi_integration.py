"""End-to-end smoke test of the optimization step.

Fits GPs with the production fitting routine on a small synthetic dataset,
then verifies that qNEHVI under the linear urea constraint proposes only
candidates that are feasible in physical units — the invariant the export
guard in run_optimization.py relies on.
"""

import numpy as np
import pytest
import torch
from botorch.models import ModelListGP
from botorch.sampling import SobolQMCNormalSampler

from config import (
    ConstraintConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizationConfig,
    get_normalized_bounds,
)
from constraints import assert_urea_feasible, get_urea_linear_constraint
from constraints.urea_dilution import CONSTRAINT_TOLERANCE
from data.preprocessing import prepare_data, standardize_reference_point
from data.transformation import build_transformer
from models import GPModel, fit_gp_model, loocv_gp_model
from acquisition import create_qnehvi_acquisition, optimize_qnehvi
from sklearn.preprocessing import StandardScaler

N_PARAMS = len(ExperimentConfig.PARAMETER_NAMES)
LB = ExperimentConfig.PARAMETER_BOUNDS[:, 0]
UB = ExperimentConfig.PARAMETER_BOUNDS[:, 1]
S = ConstraintConfig.SOLUBILIZATION_UREA
DIL_IDX = ConstraintConfig.DILUTION_FACTOR_IDX
UREA_IDX = ConstraintConfig.FINAL_UREA_IDX


def _feasible_synthetic_dataset(n=10, seed=3):
    """Random feasible points with two deterministic synthetic objectives."""
    rng = np.random.default_rng(seed)
    points = []
    while len(points) < n:
        batch = rng.uniform(LB, UB, size=(4 * n, N_PARAMS))
        feasible = batch[:, UREA_IDX] * batch[:, DIL_IDX] >= S
        points.extend(batch[feasible][: n - len(points)])
    X = np.array(points)

    # Two competing smooth objectives over the parameter space
    y1 = (0.6 * np.sin(X[:, 0] / 4.0)
          + 0.3 * np.cos(X[:, 1] * 2.0)
          + 0.1 * (X[:, 3] - 9.5) ** 2)
    y2 = -(0.05 * X[:, 2]
           + 0.4 * np.sin(X[:, 4])
           - 0.2 * X[:, 1])
    return X, np.column_stack([y1, y2])


def test_fit_gp_model_fits_and_converges():
    """The scipy fitter runs on the same path every caller uses."""
    X, Y = _feasible_synthetic_dataset()
    transformer = build_transformer(ExperimentConfig)
    train_x, train_y, _ = prepare_data(X, Y, transformer)

    model, likelihood = fit_gp_model(
        train_x=train_x,
        train_y=train_y[:, 0],
        model_class=GPModel,
        noise=ModelConfig.INITIAL_NOISE_LEVEL,
    )
    assert not model.training
    assert not likelihood.training
    # Fitted hyperparameters are finite and strictly positive
    assert torch.isfinite(likelihood.noise).all()
    assert likelihood.noise > 0
    lengthscales = model.covar_module.base_kernel.lengthscale
    assert (lengthscales > 0).all()
    assert torch.isfinite(lengthscales).all()

    # The model predicts without error
    with torch.no_grad():
        posterior = model(train_x)
    assert torch.isfinite(posterior.mean).all()


def test_loocv_uses_production_fitting_regime(tmp_path):
    """LOOCV now refits through fit_gp_model (same routine as production)."""
    X, Y = _feasible_synthetic_dataset(n=5, seed=4)
    transformer = build_transformer(ExperimentConfig)
    train_x, train_y, _ = prepare_data(X, Y, transformer)

    scores = loocv_gp_model(
        train_x, train_y, objective_idx=0, path=str(tmp_path / "cv"),
        model_class=GPModel,
        scaler=StandardScaler().fit(Y[:, [0]]),
        noise=ModelConfig.INITIAL_NOISE_LEVEL,
        make_plot=False,
    )
    assert scores["rmse"] >= 0
    assert scores["mae"] >= 0
    assert 0.0 <= scores["coverage_95"] <= 1.0
    assert np.isfinite(scores["r2"])


def test_qnehvi_candidates_respect_urea_constraint():
    X, Y = _feasible_synthetic_dataset()
    transformer = build_transformer(ExperimentConfig)
    train_x, train_y, scalers = prepare_data(X, Y, transformer)

    models = []
    for i in range(Y.shape[1]):
        model, _ = fit_gp_model(
            train_x=train_x,
            train_y=train_y[:, i],
            model_class=GPModel,
            noise=ModelConfig.INITIAL_NOISE_LEVEL,
        )
        models.append(model)
    multi_model = ModelListGP(*models)

    reference_point = standardize_reference_point(
        OptimizationConfig.REFERENCE_POINT, scalers
    )
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
    acq_function = create_qnehvi_acquisition(
        model=multi_model,
        reference_point=reference_point,
        X_baseline=train_x,
        sampler=sampler,
    )

    # Reduced acquisition-optimization budget (smoke-test scale)
    candidates = optimize_qnehvi(
        acq_function=acq_function,
        bounds=get_normalized_bounds(train_x.shape[1]),
        batch_size=3,
        mc_samples=128,
        num_restarts=2,
        raw_samples=32,
        sequential=OptimizationConfig.SEQUENTIAL_OPTIMIZATION,
        inequality_constraints=[get_urea_linear_constraint()],
    )

    assert candidates.shape == (3, N_PARAMS)
    assert (candidates >= -1e-6).all()
    assert (candidates <= 1 + 1e-6).all()

    physical = transformer.unit_to_physical_model(candidates, as_tensor=True).double()
    # Raises if any candidate violates the physical constraint
    values = assert_urea_feasible(physical)
    assert (values >= -CONSTRAINT_TOLERANCE).all()
    np.testing.assert_allclose(values,
                               (physical[:, UREA_IDX] * physical[:, DIL_IDX] - S).numpy(),
                               atol=1e-12)
    assert (physical.numpy() >= LB - 1e-9).all()
    assert (physical.numpy() <= UB + 1e-9).all()
