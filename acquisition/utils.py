"""
Utilities for acquisition function optimization and experimental planning.
"""

import logging
import os
from typing import Callable

import numpy as np
import pandas as pd
import torch

from config import ConstraintConfig
from data.transformation import ParameterTransformer

logger = logging.getLogger(__name__)


def _validate_design_counts(**counts):
    for name, value in counts.items():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")


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
    # Add iteration column (on a copy, so the caller's DataFrame is unchanged)
    new_experiments = new_experiments.copy()
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


def generate_constrained_lhd(n_samples: int, bounds: torch.Tensor, transformer: ParameterTransformer,
                            dilution_idx: int = 2, urea_idx: int = 4,
                            solubilization_urea: float = 8.0,
                            seed: int = 42,
                            n_candidates: int = 100,
                            use_maximin: bool = True) -> np.ndarray:
    """Generate Latin Hypercube Design that respects the urea dilution constraint.

    Uses conditional sampling to maintain stratification while satisfying:
        final_urea * dilution_factor >= solubilization_urea

    The approach:
    1. Generate LHD for independent parameters (DTT, GSSG, pH)
    2. Generate LHD for dilution_factor, restricted to the range where a
       feasible urea exists (dilution_factor >= solubilization_urea / urea_upper)
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

    Raises:
        ValueError: If the constraint cannot be satisfied anywhere in the
            parameter box (solubilization_urea / urea_upper exceeds the
            dilution-factor upper bound).
    """
    from scipy.spatial.distance import pdist
    from scipy.stats import qmc

    _validate_design_counts(n_samples=n_samples, n_candidates=n_candidates)
    use_maximin = use_maximin and n_samples > 1
    rng = np.random.default_rng(seed)
    n_dims = bounds.shape[1]

    # Convert bounds to numpy for easier manipulation
    bounds_np = bounds.numpy() if isinstance(bounds, torch.Tensor) else bounds

    # Get bounds for urea
    urea_lower = bounds_np[0, urea_idx]
    urea_upper = bounds_np[1, urea_idx]

    # A dilution factor admits a feasible urea only if
    # solubilization_urea / dilution_factor <= urea_upper. If no dilution
    # factor in the box satisfies that, the design problem is infeasible.
    dilution_lower, dilution_upper = bounds_np[0, dilution_idx], bounds_np[1, dilution_idx]
    min_feasible_dilution = max(dilution_lower, solubilization_urea / urea_upper)
    if min_feasible_dilution > dilution_upper:
        raise ValueError(
            "The urea constraint cannot be satisfied within the parameter "
            f"bounds: it requires dilution_factor >= {solubilization_urea / urea_upper:.4f}, "
            f"but the dilution factor is bounded by [{dilution_lower}, {dilution_upper}]. "
            "Widen the dilution or urea bounds, or lower solubilization_urea."
        )

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

        # Generate LHD for dilution factor in user-unit space over the
        # feasible subrange [min_feasible_dilution, upper bound], so every
        # draw has a feasible urea interval and stays in bounds
        z0 = float(
            transformer.physical_to_unit_user(
                np.array([min_feasible_dilution]), cols=[dilution_idx]
            )[0]
        )
        sampler_dil = qmc.LatinHypercube(d=1, seed=rng.integers(2**31))
        samples_dil_unit = sampler_dil.random(n=n_samples).flatten()
        samples_dil_user = np.clip(z0 + samples_dil_unit * (1.0 - z0), 0.0, 1.0)
        samples_dil = transformer.unit_to_physical_user(samples_dil_user, cols=[dilution_idx])

        # For each dilution factor, compute feasible urea range and sample
        # from it. By construction min_feasible_urea <= urea_upper here.
        # Constraint: final_urea >= solubilization_urea / dilution_factor
        min_feasible_urea = np.maximum(solubilization_urea / samples_dil, urea_lower)

        # Generate stratified samples for urea within feasible ranges
        # Use Latin Hypercube approach: divide each range into n equal parts
        sampler_urea = qmc.LatinHypercube(d=1, seed=rng.integers(2**31))
        samples_urea_unit = sampler_urea.random(n=n_samples).flatten()

        # Transform unit samples to feasible ranges in user unit space
        min_feasible_urea_user = transformer.physical_to_unit_user(
            min_feasible_urea, cols=[urea_idx]
        )
        urea_upper_user = float(
            transformer.physical_to_unit_user(np.array([urea_upper]), cols=[urea_idx])[0]
        )
        samples_urea_user = samples_urea_unit * (urea_upper_user - min_feasible_urea_user) + min_feasible_urea_user
        samples_urea = transformer.unit_to_physical_user(samples_urea_user, cols=[urea_idx])

        # Combine all samples
        samples = np.zeros((n_samples, n_dims))
        samples[:, dilution_idx] = samples_dil
        samples[:, urea_idx] = samples_urea

        if independent_idx:
            samples[:, independent_idx] = samples_ind

        return samples

    if use_maximin:
        logger.info(f"Optimizing constrained LHD using maximin criterion with {n_candidates} candidates")

        best_min_dist = -np.inf
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


def generate_feasible_coverage(n_samples, bounds, transformer, dilution_idx, urea_idx,
                              solubilization_urea=8.0, seed=42, n_candidates=100,
                              use_maximin=True):
    """Cover the feasible urea region in normalized user-space coordinates.

    Filter scrambled Sobol pools by physical feasibility, avoiding the unequal
    feasible-volume weighting of conditional LHD. Each greedy design adds the
    pool point farthest from its existing experiments. Among seeded starts,
    refine cluster centers onto feasible pool points, and retain the design
    with the smallest maximum pool-to-design distance (then mean squared
    distance). This is a finite-pool coverage heuristic, not a
    globally optimal design or a strict Latin hypercube. Selected points need
    not be uniformly distributed and may favor boundaries.

    n_candidates controls greedy starts. With use_maximin=False, return the
    first n_samples feasible pool points without coverage selection.
    """
    from scipy.spatial.distance import cdist
    from scipy.stats import qmc

    _validate_design_counts(n_samples=n_samples, n_candidates=n_candidates)
    box = bounds.detach().cpu().numpy() if torch.is_tensor(bounds) else np.asarray(bounds)
    if not np.array_equal(box, transformer.get_physical_bounds()):
        raise ValueError("Design bounds must match the parameter transformer")
    if dilution_idx == urea_idx or not all(0 <= i < box.shape[1] for i in (dilution_idx, urea_idx)):
        raise ValueError("Distinct valid dilution and urea indices are required")
    if (not np.isfinite(solubilization_urea) or solubilization_urea < 0
            or box[0, dilution_idx] <= 1 or box[0, urea_idx] < 0):
        raise ValueError("Urea coverage requires nonnegative urea and dilution factors greater than one")
    if box[1, dilution_idx] * box[1, urea_idx] <= solubilization_urea:
        raise ValueError("The urea constraint has no positive-volume feasible region within the bounds")

    pool_target = max(2048, 64 * n_samples)
    exponent = int(np.ceil(np.log2(pool_target)))
    rng = np.random.default_rng(seed)
    chunks = []
    accepted = 0
    for _ in range(100):
        sampler = qmc.Sobol(d=box.shape[1], scramble=True, seed=int(rng.integers(2**31)))
        unit = sampler.random_base2(exponent)
        physical = transformer.unit_to_physical_user(unit)
        feasible = physical[:, dilution_idx] * physical[:, urea_idx] >= solubilization_urea
        chunks.append(unit[feasible])
        accepted += int(feasible.sum())
        if accepted >= pool_target:
            break
    if accepted < pool_target:
        raise RuntimeError(
            f"Only {accepted} feasible pool points found; need {pool_target} for coverage selection. "
            "The feasible region may be too narrow; use constrained_lhd or revise the bounds."
        )
    pool = np.vstack(chunks)[:pool_target]
    if not use_maximin:
        return transformer.unit_to_physical_user(pool[:n_samples], as_tensor=True)

    best_score = (np.inf, np.inf)
    best_indices = None
    starts = rng.choice(len(pool), size=min(n_candidates, len(pool)), replace=False)
    for start in starts:
        indices = [int(start)]
        distances = np.sum((pool - pool[start]) ** 2, axis=1)
        for _ in range(1, n_samples):
            next_index = int(np.argmax(distances))
            indices.append(next_index)
            distances = np.minimum(distances, np.sum((pool - pool[next_index]) ** 2, axis=1))
        score = (float(distances.max()), float(distances.mean()))
        if score < best_score:
            best_score, best_indices = score, indices
        # Farthest-point selection tends toward edges. Relocate each cluster's
        # center to its nearest feasible pool point to also cover the interior.
        # Keep the greedy design as an option if refinement worsens coverage.
        for _ in range(5):
            labels = cdist(pool, pool[indices], metric="sqeuclidean").argmin(axis=1)
            centers = np.array([pool[labels == i].mean(axis=0) for i in range(n_samples)])
            refined = cdist(centers, pool, metric="sqeuclidean").argmin(axis=1).tolist()
            if len(set(refined)) != n_samples or refined == indices:
                break
            indices = refined
            nearest = cdist(pool, pool[indices], metric="sqeuclidean").min(axis=1)
            score = (float(nearest.max()), float(nearest.mean()))
            if score < best_score:
                best_score, best_indices = score, indices
    logger.info("Feasible coverage: %d pool points, estimated covering radius %.4f in user-unit space",
                len(pool), np.sqrt(best_score[0]))
    return transformer.unit_to_physical_user(pool[best_indices], as_tensor=True)


def generate_initial_design(n_samples: int, bounds: torch.Tensor, transformer: ParameterTransformer,
                          seed: int = 42,
                          n_candidates: int = 100,
                          use_maximin: bool = True,
                          design_strategy: str = "lhs",
                          constraint_callable: Callable = None,
                          oversampling_factor: int = 10,
                          solubilization_urea: float = 8.0,
                          dilution_idx: int = None,
                          urea_idx: int = None)-> torch.Tensor:
    """Generate an initial design with explicit sampling and constraint strategy.

    Args:
        n_samples: Number of initial samples to generate.
        bounds: Parameter bounds (2 x d tensor).
        transformer: Parameter transformer object.
        seed: Random seed for reproducibility.
        n_candidates: Number of candidate LHD designs or feasible-coverage starts.
        use_maximin: Whether to apply design selection (coverage or maximin).
        design_strategy: How to respect constraints:
                        - "feasible_coverage": Sobol feasible pool and greedy
                          coverage selection in normalized user space.
                        - "lhs": plain Latin hypercube over the full box.
                        - "constrained_lhd": specialized design for the urea
                          dilution constraint that preserves stratification.
                        - "rejection": sample the full box and keep only points
                          with nonnegative ``constraint_callable`` values.
        constraint_callable: Callable that takes a tensor of physical user-unit
                        samples and returns nonnegative values for feasible
                        samples. Required for the "rejection" strategy;
                        ignored otherwise.
        oversampling_factor: Factor by which to oversample when using rejection sampling.
        solubilization_urea: Urea concentration in solubilization buffer (M) for constrained LHD.
        dilution_idx: Index of the dilution factor for the constrained LHD.
                      Defaults to ConstraintConfig.DILUTION_FACTOR_IDX.
        urea_idx: Index of the final urea for the constrained LHD.
                  Defaults to ConstraintConfig.FINAL_UREA_IDX.

    Returns:
        Initial design samples (n_samples x d).

    Raises:
        ValueError: For an unknown strategy, or when a constraint callable is
            supplied while the strategy would ignore it.
    """
    from scipy.spatial.distance import pdist
    from scipy.stats import qmc

    _validate_design_counts(
        n_samples=n_samples, n_candidates=n_candidates, oversampling_factor=oversampling_factor
    )
    use_maximin = use_maximin and n_samples > 1
    if design_strategy not in ("lhs", "constrained_lhd", "rejection", "feasible_coverage"):
        raise ValueError(
            f"Unknown design_strategy {design_strategy!r}; expected 'lhs', "
            "'constrained_lhd', 'feasible_coverage', or 'rejection'."
        )
    if design_strategy == "lhs" and constraint_callable is not None:
        raise ValueError(
            "A constraint_callable was supplied but design_strategy='lhs' "
            "would ignore it. Pass design_strategy='rejection' (generic) or "
            "'constrained_lhd' (urea dilution constraint)."
        )
    if design_strategy == "rejection" and constraint_callable is None:
        raise ValueError("design_strategy='rejection' requires a constraint_callable.")

    if design_strategy == "feasible_coverage":
        if constraint_callable is not None:
            raise ValueError("feasible_coverage uses the explicit urea parameters, not a constraint_callable")
        return generate_feasible_coverage(
            n_samples, bounds, transformer,
            ConstraintConfig.DILUTION_FACTOR_IDX if dilution_idx is None else dilution_idx,
            ConstraintConfig.FINAL_UREA_IDX if urea_idx is None else urea_idx,
            solubilization_urea, seed, n_candidates, use_maximin,
        )

    if design_strategy == "constrained_lhd":
        # Use constrained LHD that maintains stratification
        logger.info("Using constrained LHD for urea dilution constraint (preserves stratification)")

        samples = generate_constrained_lhd(
            n_samples=n_samples,
            bounds=bounds,
            transformer=transformer,
            dilution_idx=(
                ConstraintConfig.DILUTION_FACTOR_IDX if dilution_idx is None else dilution_idx
            ),
            urea_idx=(
                ConstraintConfig.FINAL_UREA_IDX if urea_idx is None else urea_idx
            ),
            solubilization_urea=solubilization_urea,
            seed=seed,
            n_candidates=n_candidates,
            use_maximin=use_maximin
        )

        return torch.from_numpy(samples).double()

    # Create Latin Hypercube sampler for the plain and rejection strategies
    sampler = qmc.LatinHypercube(d=bounds.shape[1], seed=seed)

    if design_strategy == "rejection":
        # Use rejection sampling to ensure constraint satisfaction
        logger.info(f"Using rejection sampling with constraint (oversampling_factor={oversampling_factor})")

        n_total_needed = n_samples * oversampling_factor
        all_samples_unit = []
        n_accepted = 0
        attempts = 0
        max_attempts = 100

        while n_accepted < n_total_needed and attempts < max_attempts:
            # Generate batch of samples
            batch_size = min(n_total_needed * 2, 10000)
            batch_samples = sampler.random(n=batch_size)

            # Convert to tensor and denormalize for constraint checking
            batch_tensor = transformer.unit_to_physical_user(batch_samples, as_tensor=True)

            # Check constraint satisfaction
            constraint_values = constraint_callable(batch_tensor)
            feasible_mask = torch.isfinite(constraint_values) & (constraint_values >= 0)

            # Keep feasible samples (in unit space)
            feasible_samples_unit = batch_samples[feasible_mask.cpu().numpy()]
            all_samples_unit.append(feasible_samples_unit)
            n_accepted += len(feasible_samples_unit)

            attempts += 1
            if attempts % 20 == 0:
                logger.info(
                    f"  Rejection sampling: {len(np.vstack(all_samples_unit))} feasible "
                    f"samples after {attempts} attempts"
                )

        if n_accepted < n_samples:
            raise RuntimeError(
                f"Found only {n_accepted} feasible samples after {attempts} attempts; "
                f"requested {n_samples}."
            )

        # Combine all feasible samples
        all_samples_unit = np.vstack(all_samples_unit)
        logger.info(f"Found {len(all_samples_unit)} feasible samples")

        if use_maximin and len(all_samples_unit) > n_samples:
            # Select best spread subset using maximin criterion
            logger.info(f"Selecting {n_samples} samples with best spread...")

            # Seeded so the whole design is reproducible from `seed`
            subset_rng = np.random.default_rng(seed)
            best_min_dist = -np.inf
            best_indices = None

            for _ in range(n_candidates):
                # Random subset selection
                indices = subset_rng.choice(
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

        best_min_dist = -np.inf
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
    samples = transformer.unit_to_physical_user(samples_unit, as_tensor=True)

    logger.info(f"Generated initial design with {n_samples} samples")
    return samples
