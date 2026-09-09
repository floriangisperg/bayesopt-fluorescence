"""
Gaussian Process model fitting and loading utilities.

Provides functions for training GPs, and saving/loading model states.
Saved models carry a fingerprint of their training data and configuration,
which is verified on load so data or config drift cannot silently produce
wrong predictions.
"""

import hashlib
import json
import logging
import os
import re
from typing import List, Tuple

import gpytorch
import torch
from botorch.fit import fit_gpytorch_mll

from config import ExperimentConfig, ModelConfig

logger = logging.getLogger(__name__)


def _tensor_sha1(tensor: torch.Tensor) -> str:
    """Content hash of a tensor, stable across runs for identical data."""
    return hashlib.sha1(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def _config_snapshot() -> dict:
    """The experiment-configuration state a GP was trained under.

    Everything that changes how training inputs are mapped into model space
    or how the kernel is built. Bounds and transformations shape the unit
    space; KERNEL_NU selects the kernel family.
    """
    return {
        "parameter_bounds": ExperimentConfig.PARAMETER_BOUNDS.tolist(),
        "parameter_transformation": ExperimentConfig.PARAMETER_TRANSFORMATION,
        "kernel_nu": ModelConfig.KERNEL_NU,
    }


def _config_sha1() -> str:
    canonical = json.dumps(_config_snapshot(), sort_keys=True)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


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
                  train_y_standardized: torch.Tensor, objective_idx: int = 0,
                  strict: bool = True):
    """Load a Gaussian Process model and its likelihood from file.

    The model is reconstructed with the training data passed in here (GPyTorch
    keeps it outside the state dict), so a fingerprint of the data and config
    it was originally trained on is verified to catch drift: loading with
    different data or a changed configuration would silently pair stale
    hyperparameters with a different problem.

    Args:
        filepath: Path to the saved model file.
        model_class: GP model class to be instantiated.
        train_x_normalized: Normalized training inputs.
        train_y_standardized: Standardized training outputs.
        objective_idx: Index of the objective to load (for multi-output models).
        strict: Raise on fingerprint mismatch. With ``strict=False`` a
                mismatch is logged as a warning and the model still loads.

    Returns:
        Tuple of (model, likelihood).

    Raises:
        FileNotFoundError: If model file is not found.
        ValueError: If saved file format is invalid, or (with ``strict=True``)
            the training data or configuration does not match the checkpoint.
    """
    try:
        saved_data = torch.load(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {filepath}")

    # Validate saved data structure
    if 'model_state_dict' not in saved_data or 'likelihood_state_dict' not in saved_data:
        raise ValueError(f"Invalid model file format: {filepath}")

    # Verify the checkpoint belongs to the data and config at hand
    fingerprint = saved_data.get('training_data_fingerprint')
    if fingerprint is None:
        logger.info(
            f"No training fingerprint stored in {filepath} (legacy "
            "checkpoint); skipping data/config verification."
        )
    else:
        mismatches = []
        if fingerprint.get('train_x_sha1') != _tensor_sha1(train_x_normalized):
            mismatches.append("the training inputs differ")
        if fingerprint.get('train_y_sha1') != _tensor_sha1(
                train_y_standardized[:, objective_idx]):
            mismatches.append("the training targets differ")
        if fingerprint.get('config_sha1') != _config_sha1():
            mismatches.append(
                "the experiment configuration differs (bounds, parameter "
                "transformations, or kernel settings)"
            )
        if mismatches:
            message = (
                f"Checkpoint {filepath} does not match the data at hand: "
                + " and ".join(mismatches)
                + ". Retrain the models on the current data, or pass "
                "strict=False to load anyway."
            )
            if strict:
                raise ValueError(message)
            logger.warning(message)

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

    Stores a fingerprint of the training data and experiment configuration
    alongside the state dicts, so ``load_gp_model`` can detect data or config
    drift before the model is used.

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
        },
        'training_data_fingerprint': {
            'train_x_sha1': _tensor_sha1(model.train_inputs[0]),
            'train_y_sha1': _tensor_sha1(model.train_targets),
            'config_sha1': _config_sha1(),
        },
        'config_snapshot': _config_snapshot(),
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
