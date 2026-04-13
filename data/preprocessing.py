"""
Data preprocessing utilities for protein refolding optimization.

Provides functions for normalizing experimental parameters and standardizing
objective values for Gaussian Process modeling.
"""

import os
import logging
from typing import Tuple, List

import numpy as np
import torch
import pickle
from sklearn.preprocessing import StandardScaler

from data.transformation import ParameterTransformer

logger = logging.getLogger(__name__)


def normalize_parameters(X: np.ndarray, bounds: np.ndarray) -> torch.Tensor:
    """Normalize experimental parameters to [0,1] range.

    Args:
        X: Raw experimental parameters (n_samples x n_features).
        bounds: Parameter bounds (n_features x 2) with [lower, upper].

    Returns:
        Normalized parameters as torch tensor.
    """
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]

    X_normalized = (X - lower_bounds) / (upper_bounds - lower_bounds)

    return torch.from_numpy(X_normalized).double()


def standardize_objectives(y: np.ndarray) -> Tuple[torch.Tensor, List[StandardScaler]]:
    """Standardize objective values to zero mean and unit variance.

    Args:
        y: Raw objective values (n_samples x n_objectives).

    Returns:
        Tuple of (standardized objectives, list of scalers for inverse transform).
    """
    scalers = []
    y_standardized = []

    n_objectives = y.shape[1]
    for i in range(n_objectives):
        scaler = StandardScaler()
        y_i_standardized = scaler.fit_transform(y[:, i].reshape(-1, 1)).flatten()
        y_standardized.append(y_i_standardized)
        scalers.append(scaler)

    y_standardized = np.column_stack(y_standardized)
    return torch.from_numpy(y_standardized).double(), scalers


def prepare_data(X: np.ndarray, y: np.ndarray, transformer: ParameterTransformer) -> Tuple[torch.Tensor, torch.Tensor, List]:
    """Prepare training data for GP modeling.

    Args:
        X: Raw experimental parameters.
        y: Raw objective values.
        transformer: Parameter transformer object.

    Returns:
        Tuple of (normalized X, standardized y, scalers).
    """
    logger.info(f"Preparing data: X shape {X.shape}, y shape {y.shape}")

    # Normalize parameters
    X_normalized = transformer.physical_to_unit_model(X, as_tensor=True)

    # Standardize objectives
    y_standardized, scalers = standardize_objectives(y)

    logger.info("Data preparation completed successfully")
    return X_normalized, y_standardized, scalers


def save_scalers(scalers: List, filepath: str):
    """Save objective scalers to file.

    Args:
        scalers: List of StandardScaler objects.
        filepath: Path to save the scalers.
    """
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(filepath, 'wb') as f:
        pickle.dump(scalers, f)

    logger.info(f"Saved {len(scalers)} scalers to {filepath}")


def load_scalers(filepath: str) -> List:
    """Load objective scalers from file.

    Args:
        filepath: Path to the saved scalers.

    Returns:
        List of StandardScaler objects.
    """
    with open(filepath, 'rb') as f:
        scalers = pickle.load(f)

    logger.info(f"Loaded {len(scalers)} scalers from {filepath}")
    return scalers


def inverse_transform_objectives(y_standardized: torch.Tensor, scalers: List) -> np.ndarray:
    """Inverse transform standardized objectives to original scale.

    Args:
        y_standardized: Standardized objective values.
        scalers: List of StandardScaler objects.

    Returns:
        Objectives in original scale.
    """
    y_numpy = y_standardized.detach().cpu().numpy()
    y_original = []

    n_objectives = y_numpy.shape[1]
    for i in range(n_objectives):
        y_i_original = scalers[i].inverse_transform(y_numpy[:, i].reshape(-1, 1)).flatten()
        y_original.append(y_i_original)

    return np.column_stack(y_original)