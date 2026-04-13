"""
Utilities for acquisition function optimization and experimental planning.
"""

import os
import logging
from typing import List, Callable

import numpy as np
import pandas as pd
import torch

from data.transformation import ParameterTransformer

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


def generate_constrained_lhd(n_samples: int, bounds: torch.Tensor, transformer: ParameterTransformer,
                            dilution_idx: int = 2, urea_idx: int = 4,
                            solubilization_urea: float = 8.0,
                            seed: int = 42,
                            n_candidates: int = 100,
                            use_maximin: bool = True) -> np.ndarray:
    """Generate Latin Hypercube Design that respects the urea dilution constraint.

    Uses conditional sampling to maintain stratification while satisfying:
        final_urea * dilution_factor > solubilization_urea

    The approach:
    1. Generate LHD for independent parameters (DTT, GSSG, pH)
    2. Generate LHD for dilution_factor
    3. For each dilution_factor, sample final_urea from its feasible range

    Args:
        n_samples: Number of samples to generate.
        bounds: Parameter bounds (2 x d tensor).
        transformer: Parameter transformer object.
        dilution_idx: Index of dilution factor parameter.
        urea_idx: Index of final urea parameter.
        solubilization_urea: Urea concentration in solubilization buffer (M).
        seed: Random seed for reproducibility.
        n_candidates: Number of candidate designs for maximin optimization.
        use_maximin: Whether to apply maximin criterion optimization.

    Returns:
        Array of samples (n_samples x d) satisfying the constraint.
    """
    from scipy.stats import qmc
    from scipy.spatial.distance import pdist

    rng = np.random.default_rng(seed)
    n_dims = bounds.shape[1]

    # Convert bounds to numpy for easier manipulation
    bounds_np = bounds.numpy() if isinstance(bounds, torch.Tensor) else bounds

    # Get bounds for urea
    urea_lower = bounds_np[0, urea_idx]
    urea_upper = bounds_np[1, urea_idx]

    # Indices of independent parameters
    independent_idx = [i for i in range(n_dims) if i not in [dilution_idx, urea_idx]]

    def generate_single_design():
        """Generate a single constrained LHD."""
        # Generate LHD for independent parameters
        if independent_idx:
            sampler_ind = qmc.LatinHypercube(d=len(independent_idx), seed=rng.integers(2**31))
            samples_ind_unit = sampler_ind.random(n=n_samples)

            # Denormalize independent parameters to their actual bounds
            samples_ind = transformer.unit_to_physical_user(samples_ind_unit, cols=independent_idx)

        # Generate LHD for dilution factor (in unit space)
        sampler_dil = qmc.LatinHypercube(d=1, seed=rng.integers(2**31))
        samples_dil_unit = sampler_dil.random(n=n_samples).flatten()
        samples_dil = transformer.unit_to_physical_user(samples_dil_unit, cols=[dilution_idx])

        # For each dilution factor, compute feasible urea range and sample from it
        # Constraint: final_urea > solubilization_urea / dilution_factor
        min_feasible_urea = solubilization_urea / samples_dil

        # Clip to parameter bounds
        min_feasible_urea = np.maximum(min_feasible_urea, urea_lower)

        # Check if all samples are feasible
        feasible = min_feasible_urea <= urea_upper
        if not np.all(feasible):
            logger.warning(f"Some samples have no feasible urea range (dilution too low)")

        # Generate stratified samples for urea within feasible ranges
        # Use Latin Hypercube approach: divide each range into n equal parts
        sampler_urea = qmc.LatinHypercube(d=1, seed=rng.integers(2**31))
        samples_urea_unit = sampler_urea.random(n=n_samples).flatten()

        # Transform unit samples to feasible ranges
        samples_urea = np.zeros(n_samples)
        for i in range(n_samples):
            urea_min = min_feasible_urea[i]
            samples_urea[i] = samples_urea_unit[i] * (urea_upper - urea_min) + urea_min

        # Combine all samples
        samples = np.zeros((n_samples, n_dims))
        samples[:, dilution_idx] = samples_dil
        samples[:, urea_idx] = samples_urea

        if independent_idx:
            samples[:, independent_idx] = samples_ind

        return samples

    if use_maximin:
        logger.info(f"Optimizing constrained LHD using maximin criterion with {n_candidates} candidates")

        best_min_dist = 0
        best_samples = None

        for _ in range(n_candidates):
            candidate_samples = generate_single_design()

            # Calculate minimum pairwise distance in unit space for fair comparison
            # Normalize to unit space
            samples_unit = transformer.physical_to_unit_user(candidate_samples)
            min_dist = pdist(samples_unit).min()

            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_samples = candidate_samples

        samples = best_samples
        logger.info(f"Best constrained design has minimum distance: {best_min_dist:.4f}")
    else:
        samples = generate_single_design()

    return samples


def generate_initial_design(n_samples: int, bounds: torch.Tensor, transformer: ParameterTransformer,
                          seed: int = 42,
                          n_candidates: int = 100,
                          use_maximin: bool = True,
                          constraint_callable: Callable = None,
                          oversampling_factor: int = 10,
                          solubilization_urea: float = 8.0)-> torch.Tensor:
    """Generate initial experimental design using Latin Hypercube Sampling.

    Supports constraint satisfaction via:
    - Constrained LHD for urea constraint (maintains stratification)
    - Rejection sampling for other constraints

    Args:
        n_samples: Number of initial samples to generate.
        bounds: Parameter bounds (2 x d tensor).
        transformer: Parameter transformer object.
        seed: Random seed for reproducibility.
        n_candidates: Number of candidate designs to evaluate for maximin criterion.
        use_maximin: Whether to apply maximin criterion optimization.
        constraint_callable: Optional callable that takes a tensor and returns positive
                        values for feasible samples.
        oversampling_factor: Factor by which to oversample when using rejection sampling.
        solubilization_urea: Urea concentration in solubilization buffer (M) for constrained LHD.

    Returns:
        Initial design samples (n_samples x d).
    """
    from scipy.stats import qmc
    from scipy.spatial.distance import pdist

    # Check if this is the urea constraint - use specialized constrained LHD
    # Check by function name or module
    is_urea_constraint = False
    if constraint_callable is not None:
        func_name = getattr(constraint_callable, '__name__', '')
        func_module = getattr(constraint_callable, '__module__', '')
        is_urea_constraint = 'urea' in func_name.lower() or 'urea' in func_module.lower()

    if constraint_callable is not None and is_urea_constraint:
        # Use constrained LHD that maintains stratification
        logger.info("Using constrained LHD for urea dilution constraint (preserves stratification)")

        samples = generate_constrained_lhd(
            n_samples=n_samples,
            bounds=bounds,
            transformer=transformer,
            dilution_idx=2, #TODO: elimate hardcode idx, anzahl an constrained idx
            urea_idx=4,
            solubilization_urea=solubilization_urea,
            seed=seed,
            n_candidates=n_candidates,
            use_maximin=use_maximin
        )

        return torch.from_numpy(samples).double()

    # Create Latin Hypercube sampler for non-urea constraints or no constraint
    sampler = qmc.LatinHypercube(d=bounds.shape[1], seed=seed)

    if constraint_callable is not None:
        # Use rejection sampling to ensure constraint satisfaction
        logger.info(f"Using rejection sampling with constraint (oversampling_factor={oversampling_factor})")

        n_total_needed = n_samples * oversampling_factor
        all_samples_unit = []
        attempts = 0
        max_attempts = 100

        while len(all_samples_unit) < n_total_needed and attempts < max_attempts:
            # Generate batch of samples
            batch_size = min(n_total_needed * 2, 10000)
            batch_samples = sampler.random(n=batch_size)

            # Convert to tensor and denormalize for constraint checking
            batch_tensor = transformer.unit_to_physical_user(batch_samples)

            # Check constraint satisfaction
            constraint_values = constraint_callable(batch_tensor)
            feasible_mask = constraint_values > 0

            # Keep feasible samples (in unit space)
            feasible_samples_unit = batch_samples[feasible_mask.numpy()]
            all_samples_unit.append(feasible_samples_unit)

            attempts += 1
            if attempts % 20 == 0:
                logger.info(f"  Rejection sampling: {len(np.vstack(all_samples_unit))} feasible samples after {attempts} attempts")

        if len(all_samples_unit) == 0:
            raise RuntimeError("Could not find any feasible samples satisfying the constraint!")

        # Combine all feasible samples
        all_samples_unit = np.vstack(all_samples_unit)
        logger.info(f"Found {len(all_samples_unit)} feasible samples")

        if use_maximin and len(all_samples_unit) > n_samples:
            # Select best spread subset using maximin criterion
            logger.info(f"Selecting {n_samples} samples with best spread...")

            best_min_dist = 0
            best_indices = None

            for _ in range(n_candidates):
                # Random subset selection
                indices = np.random.choice(
                    len(all_samples_unit), size=n_samples, replace=False
                )
                candidate_subset = all_samples_unit[indices]

                # Calculate minimum pairwise distance
                min_dist = pdist(candidate_subset).min()

                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_indices = indices

            samples_unit = all_samples_unit[best_indices]
            logger.info(f"Best design has minimum distance: {best_min_dist:.4f}")
        else:
            # Just take first n_samples
            samples_unit = all_samples_unit[:n_samples]

    elif use_maximin and n_samples <= 100:
        # Original maximin optimization without constraints
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
    samples = transformer.unit_to_physical_user(samples_unit)

    logger.info(f"Generated initial design with {n_samples} samples")
    return samples