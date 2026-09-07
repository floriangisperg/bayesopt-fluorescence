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


def standardize_reference_point(ref_point: List[float], scalers: List) -> torch.Tensor:
    """Map a reference point from real objective units to standardized space.

    qNEHVI operates on the standardized objectives the GP models are trained
    on, while ``OptimizationConfig.REFERENCE_POINT`` is specified in real
    (measured) objective units. This applies each objective's scaler so the
    reference point can be passed to the acquisition function.

    Args:
        ref_point: Reference point in real units, ordered like the objectives.
        scalers: Fitted StandardScalers, one per objective.

    Returns:
        Reference point in standardized space as a float64 tensor.
    """
    if len(ref_point) != len(scalers):
        raise ValueError(
            f"Reference point has {len(ref_point)} values but there are "
            f"{len(scalers)} objective scalers"
        )

    standardized = []
    for value, scaler in zip(ref_point, scalers):
        transformed = scaler.transform(
            np.array([[value]], dtype=np.float64)
        )
        standardized.append(float(transformed[0, 0]))

    return torch.tensor(standardized, dtype=torch.float64)


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