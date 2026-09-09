"""
Gaussian Process model fitting and loading utilities.

Provides functions for training GPs, and saving/loading model states.
"""

import os
import re
import logging
from typing import Tuple, List

import gpytorch
import torch
from botorch.fit import fit_gpytorch_mll

logger = logging.getLogger(__name__)


def sort_objective_files(filenames: List[str]) -> List[str]:
    """Sort saved model/scaler files by their embedded objective index.

    ``train_models.py`` names files ``model_{i}_{objective}.pth`` and
    ``scaler_{i}_{objective}.pkl``. Plain lexicographic sorting would place
    ``model_10_...`` before ``model_2_...``, mispairing files with objectives
    once more than nine objectives exist.

    Args:
        filenames: File names to sort.

    Returns:
        File names ordered by objective index (lexicographically for names
        without an index).
    """
    def key(name: str):
        match = re.search(r'_(\d+)_', name)
        return (int(match.group(1)), name) if match else (float('inf'), name)

    return sorted(filenames, key=key)


def load_gp_model(filepath: str, model_class, train_x_normalized: torch.Tensor,
                  train_y_standardized: torch.Tensor, objective_idx: int = 0):
    """Load a Gaussian Process model and its likelihood from file.

    Args:
        filepath: Path to the saved model file.
        model_class: GP model class to be instantiated.
        train_x_normalized: Normalized training inputs.
        train_y_standardized: Standardized training outputs.
        objective_idx: Index of the objective to load (for multi-output models).

    Returns:
        Tuple of (model, likelihood).

    Raises:
        FileNotFoundError: If model file is not found.
        ValueError: If saved file format is invalid.
    """
    try:
        saved_data = torch.load(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {filepath}")

    # Validate saved data structure
    if 'model_state_dict' not in saved_data or 'likelihood_state_dict' not in saved_data:
        raise ValueError(f"Invalid model file format: {filepath}")

    # Create fresh likelihood object
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    # Extract single objective from multi-output training data
    train_y_single = train_y_standardized[:, objective_idx]

    # Create model with fresh likelihood
    model = model_class(train_x_normalized, train_y_single, likelihood)

    # Load both model and likelihood states
    model.load_state_dict(saved_data['model_state_dict'])
    likelihood.load_state_dict(saved_data['likelihood_state_dict'])

    # Set to evaluation mode
    model.eval()
    likelihood.eval()

    logger.info(f'Model and likelihood loaded successfully from {filepath}')
    logger.info(f'Model class: {saved_data.get("model_class", "Unknown")}')
    logger.info(f'Likelihood class: {saved_data.get("likelihood_class", "Unknown")}')

    return model, likelihood


def save_gp_model(model, likelihood, filepath: str):
    """Save a Gaussian Process model and its likelihood to file.

    Args:
        model: Trained GP model.
        likelihood: Trained likelihood.
        filepath: Complete file path where model will be saved.
    """
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save model and likelihood together with metadata
    torch.save({
        'model_state_dict': model.state_dict(),
        'likelihood_state_dict': likelihood.state_dict(),
        'model_class': model.__class__.__name__,
        'likelihood_class': likelihood.__class__.__name__,
        'training_data_shape': {
            'train_x_shape': model.train_inputs[0].shape if model.train_inputs else None,
            'train_y_shape': model.train_targets.shape if hasattr(model, 'train_targets') else None
        }
    }, filepath)

    logger.info(f'Model and likelihood saved successfully to {filepath}')


def fit_gp_model(train_x: torch.Tensor, train_y: torch.Tensor, model_class,
                 noise: float = 0.01) -> Tuple[object, object]:
    """Fit a Gaussian Process model to training data.

    Hyperparameters are optimized by maximizing the exact marginal
    log-likelihood with BoTorch's ``fit_gpytorch_mll`` (scipy L-BFGS-B, with
    random restarts on failure). The optimizer runs to convergence, so there
    is no learning rate or iteration budget to tune, and every caller —
    model training and LOOCV alike — fits under the identical regime.

    Args:
        train_x: Input features.
        train_y: Target outputs.
        model_class: GP model class to be instantiated.
        noise: Initial noise level for the likelihood (starting point for the
               optimizer; the fitted value may differ).

    Returns:
        Tuple of (model, likelihood).
    """
    # Initialize likelihood (noise is learnable during training)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = noise
    model = model_class(train_x, train_y, likelihood)

    # Exact marginal log-likelihood objective
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Cap the optimizer budget in smoke-test mode; on the small datasets this
    # workflow uses, the full fit converges quickly anyway.
    optimizer_kwargs = None
    if os.environ.get("SMOKE_TEST"):
        optimizer_kwargs = {"options": {"maxiter": 100}}

    fit_gpytorch_mll(mll, optimizer_kwargs=optimizer_kwargs)

    # Set to evaluation mode
    model.eval()
    likelihood.eval()

    return model, likelihood
