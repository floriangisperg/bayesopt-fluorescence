"""
Gaussian Process model fitting and loading utilities.

Provides functions for training GP models, saving/loading model states,
and visualizing training progress.
"""

import os
import logging
from typing import Tuple, List

import gpytorch
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


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
                 noise: float = 0.01, num_train_iters: int = 1000, lr: float = 0.01,
                 save_model: bool = False, filepath: str = None) -> Tuple[object, object, List[float]]:
    """Fit a Gaussian Process model to training data.

    Args:
        train_x: Input features.
        train_y: Target outputs.
        model_class: GP model class to be instantiated.
        noise: Initial noise level for the likelihood.
        num_train_iters: Number of training iterations.
        lr: Learning rate for the optimizer.
        save_model: Whether to save the model after training.
        filepath: File path for saving the model.

    Returns:
        Tuple of (model, likelihood, losses).
    """
    # Initialize likelihood (noise is learnable during training)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = noise
    model = model_class(train_x, train_y, likelihood)

    # Set to training mode
    model.train()
    likelihood.train()

    # Optimizer and objective
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    losses = []
    for i in range(num_train_iters):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        # Ensure loss is a scalar
        if loss.dim() > 0:
            loss = loss.mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Set to evaluation mode
    model.eval()
    likelihood.eval()

    if save_model and filepath:
        save_gp_model(model, likelihood, filepath)

    return model, likelihood, losses


def plot_training_loss(losses: List[float], path: str, make_plot: bool = True):
    """Plot training loss over iterations.

    Args:
        losses: List of loss values during training.
        path: Base path for saving the plot.
        make_plot: Whether to save the plot to file.
    """
    plt.figure(figsize=(7.5, 2.5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('GP Training Loss Over Iterations')
    plt.legend()

    if make_plot:
        plt.savefig(f"{path}_training_loss.png", dpi=300)
    plt.close()


def plot_predictions(test_x: torch.Tensor, test_y: torch.Tensor,
                    predicted_y: torch.Tensor, path: str, make_plot: bool = False):
    """Plot predictions vs actual values.

    Args:
        test_x: Test inputs.
        test_y: True test outputs.
        predicted_y: Predicted outputs.
        path: Base path for saving the plot.
        make_plot: Whether to save the plot to file.
    """
    plt.figure(figsize=(7.5, 2.5))
    plt.plot(test_x, test_y, 'r*', label='Actual Data')
    plt.plot(test_x, predicted_y, 'b-', label='Predicted Data')
    plt.xlabel('Input Features')
    plt.ylabel('Output Targets')
    plt.title('Comparison of Predictions and Actual Data')
    plt.legend()

    if make_plot:
        plt.savefig(f"{path}_pred_vs_actual.png", dpi=300)
    plt.close()