"""
Gaussian Process model validation utilities.

Provides leave-one-out cross-validation for GP model assessment.
"""

import logging
from typing import List

import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def loocv_gp_model(train_x: torch.Tensor, train_y: torch.Tensor,
                   objective_idx: int, path: str, model_class, scaler, make_plot: bool = False):
    """Perform leave-one-out cross-validation on GP model.

    Args:
        train_x: Training input data.
        train_y: Training output data.
        objective_idx: Index of the objective to validate.
        path: Base path for saving plots.
        model_class: GP model class.
        scaler: Scaler used for standardization.
        make_plot: Whether to generate plots.

    Returns:
        Dictionary of validation scores.
    """
    n_samples = train_x.shape[0]
    predictions = []
    uncertainties = []
    actual_values = []

    logger.info(f"Starting LOOCV for {n_samples} samples")

    for i in range(n_samples):
        # Create training sets leaving out sample i
        mask = torch.ones(n_samples, dtype=torch.bool)
        mask[i] = False

        loo_train_x = train_x[mask]
        loo_train_y = train_y[mask][:, objective_idx]

        # Train model on LOOCV data
        from .gp_fitting import fit_gp_model
        loo_model, loo_likelihood, _ = fit_gp_model(
            loo_train_x, loo_train_y, model_class,
            num_train_iters=500  # Faster training for LOOCV
        )

        # Predict left-out sample
        loo_model.eval()
        loo_likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = train_x[i].unsqueeze(0)
            observed_pred = loo_likelihood(loo_model(test_x))
            pred_mean = observed_pred.mean.numpy()[0]
            pred_std = observed_pred.stddev.numpy()[0]

        predictions.append(pred_mean)
        uncertainties.append(pred_std)
        actual_values.append(train_y[i, objective_idx].item())

        if (i + 1) % 10 == 0:
            logger.info(f"Completed {i + 1}/{n_samples} LOOCV samples")

    # Calculate validation metrics
    predictions = np.array(predictions)
    uncertainties = np.array(uncertainties)
    actual_values = np.array(actual_values)

    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((predictions - actual_values) ** 2))

    # Mean Absolute Error
    mae = np.mean(np.abs(predictions - actual_values))

    # R² score
    ss_res = np.sum((actual_values - predictions) ** 2)
    ss_tot = np.sum((actual_values - np.mean(actual_values)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Standardized RMSE (in original scale)
    # This requires inverting the standardization
    if hasattr(scaler, 'inverse_transform'):
        pred_original = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        actual_original = scaler.inverse_transform(actual_values.reshape(-1, 1)).flatten()
        rmse_original = np.sqrt(np.mean((pred_original - actual_original) ** 2))
    else:
        rmse_original = rmse

    validation_scores = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'rmse_original_scale': rmse_original,
        'mean_uncertainty': np.mean(uncertainties),
        'coverage_95': np.mean(np.abs(predictions - actual_values) <= 1.96 * uncertainties)
    }

    logger.info(f"LOOCV Results - RMSE: {rmse:.4f}, R²: {r2:.4f}, Coverage: {validation_scores['coverage_95']:.4f}")

    if make_plot:
        plot_loocv_results(actual_values, predictions, uncertainties, path, objective_idx)

    return validation_scores


def plot_loocv_results(actual_values: np.ndarray, predictions: np.ndarray,
                      uncertainties: np.ndarray, path: str, objective_idx: int):
    """Plot LOOCV results with uncertainty bands.

    Args:
        actual_values: True values.
        predictions: Predicted values.
        uncertainties: Prediction uncertainties.
        path: Base path for saving plots.
        objective_idx: Objective index for labeling.
    """
    plt.figure(figsize=(8, 6))

    # Sort by actual values for better visualization
    sort_idx = np.argsort(actual_values)
    actual_sorted = actual_values[sort_idx]
    pred_sorted = predictions[sort_idx]
    unc_sorted = uncertainties[sort_idx]

    x_pos = range(len(actual_sorted))

    plt.errorbar(x_pos, pred_sorted, yerr=1.96 * unc_sorted,
                 fmt='o', label='Predictions with 95% CI', alpha=0.7, capsize=3)
    plt.plot(x_pos, actual_sorted, 'r*', markersize=10, label='Actual Values')

    plt.xlabel('Sample Index (sorted by actual value)')
    plt.ylabel('Standardized Value')
    plt.title(f'LOOCV Results - Objective {objective_idx + 1}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Calculate R² for display
    ss_res = np.sum((actual_sorted - pred_sorted) ** 2)
    ss_tot = np.sum((actual_sorted - np.mean(actual_sorted)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f"{path}_loocv_objective_{objective_idx + 1}.png", dpi=300)
    plt.close()

    # Also create parity plot
    plt.figure(figsize=(6, 6))
    plt.scatter(actual_values, predictions, alpha=0.7)

    # Perfect prediction line
    min_val = min(actual_values.min(), predictions.min())
    max_val = max(actual_values.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Parity Plot - Objective {objective_idx + 1}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    plt.tight_layout()
    plt.savefig(f"{path}_parity_objective_{objective_idx + 1}.png", dpi=300)
    plt.close()