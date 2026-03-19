"""
Utilities for acquisition function optimization and experimental planning.
"""

import os
import logging
from typing import List

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def save_experiments_to_excel(data: torch.Tensor, path: str) -> pd.DataFrame:
    """Convert experiment tensor to DataFrame and save as Excel file.

    Args:
        data: Tensor containing experimental parameters.
        path: Path where Excel file will be saved.

    Returns:
        DataFrame with experimental data.
    """
    # Parameter names for DataFrame
    parameter_names = [
        "DTT [mM]",
        "GSSG [mM]",
        "Dilution Factor",
        "pH",
        "Final Urea [M]"
    ]

    # Create DataFrame
    df = pd.DataFrame(data=data.numpy(), columns=parameter_names)

    # Save to Excel
    df.to_excel(path, index=False)
    logger.info(f"Saved {len(df)} experiments to {path}")

    return df


def update_experimental_database(new_experiments: pd.DataFrame,
                              iteration: int, path: str) -> pd.DataFrame:
    """Update experimental database with new experiments.

    Args:
        new_experiments: New experimental data to add.
        iteration: Current iteration number.
        path: Path to the experimental database Excel file.

    Returns:
        Updated DataFrame containing all experiments.
    """
    # Add iteration column
    new_experiments['Iteration'] = iteration

    # Check if file exists
    if os.path.exists(path):
        # Load existing data
        existing_df = pd.read_excel(path)

        # Concatenate new data
        updated_df = pd.concat([existing_df, new_experiments], ignore_index=True)
    else:
        # Create new database
        updated_df = new_experiments.copy()

    # Save updated database
    updated_df.to_excel(path, index=False)
    logger.info(f"Updated experimental database with {len(new_experiments)} new experiments")

    return updated_df


def denormalize_parameters(normalized_data: torch.Tensor,
                          bounds: torch.Tensor) -> torch.Tensor:
    """Denormalize parameters from [0,1] to original bounds.

    Args:
        normalized_data: Normalized parameter data.
        bounds: Original bounds (2 x d tensor).

    Returns:
        Denormalized parameter data.
    """
    lower_bounds = bounds[0]
    upper_bounds = bounds[1]

    return normalized_data * (upper_bounds - lower_bounds) + lower_bounds


def generate_initial_design(n_samples: int, bounds: torch.Tensor,
                          seed: int = 42,
                          n_candidates: int = 100,
                          use_maximin: bool = True) -> torch.Tensor:
    """Generate initial experimental design using Latin Hypercube Sampling.

    Args:
        n_samples: Number of initial samples to generate.
        bounds: Parameter bounds (2 x d tensor).
        seed: Random seed for reproducibility.
        n_candidates: Number of candidate designs to evaluate for maximin criterion.
        use_maximin: Whether to apply maximin criterion optimization.

    Returns:
        Initial design samples (n_samples x d).
    """
    from scipy.stats import qmc
    from scipy.spatial.distance import pdist

    # Create Latin Hypercube sampler
    sampler = qmc.LatinHypercube(d=bounds.shape[1], seed=seed)

    if use_maximin and n_samples <= 100:
        # Generate multiple candidate designs and select the one with maximin criterion
        logger.info(f"Optimizing design using maximin criterion with {n_candidates} candidates")

        best_min_dist = 0
        best_samples_unit = None

        for i in range(n_candidates):
            # Generate candidate samples in [0,1] space
            candidate_samples = sampler.random(n=n_samples)

            # Calculate minimum pairwise distance (maximin criterion)
            min_dist = pdist(candidate_samples).min()

            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_samples_unit = candidate_samples

        samples_unit = best_samples_unit
        logger.info(f"Best design has minimum distance: {best_min_dist:.4f}")
    else:
        # Generate samples in [0,1] space
        samples_unit = sampler.random(n=n_samples)

    # Denormalize to original bounds
    samples = denormalize_parameters(torch.from_numpy(samples_unit), bounds)

    logger.info(f"Generated initial design with {n_samples} samples")
    return samples