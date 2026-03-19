"""
Urea dilution constraint handling for protein refolding optimization.

Implements physical constraints for the urea dilution process to ensure
that generated experimental conditions are physically feasible.
"""

import logging
from typing import List

import numpy as np
import torch

from config import ConstraintConfig

logger = logging.getLogger(__name__)
CONSTRAINT_MARGIN = 1e-6


def check_urea_constraint(sample: np.ndarray,
                         solubilization_urea: float = None) -> bool:
    """Check if a sample satisfies the urea dilution physical constraint.

    The constraint ensures that the refolding urea concentration is positive:
    urea_refolding = (final_urea * dilution_factor - solubilization_urea) / (dilution_factor - 1) > 0

    This simplifies to: final_urea * dilution_factor > solubilization_urea

    Args:
        sample: Array containing [DTT, GSSG, dilution_factor, pH, final_urea].
        solubilization_urea: Urea concentration in solubilization buffer (M).
                           Defaults to ConstraintConfig.SOLUBILIZATION_UREA.

    Returns:
        True if the constraint is satisfied, False otherwise.
    """
    if solubilization_urea is None:
        solubilization_urea = ConstraintConfig.SOLUBILIZATION_UREA

    final_urea = sample[ConstraintConfig.FINAL_UREA_IDX]
    dilution_factor = sample[ConstraintConfig.DILUTION_FACTOR_IDX]
    urea_refolding = ((final_urea * dilution_factor) - solubilization_urea) / (dilution_factor - 1)
    return urea_refolding > 0


def iterative_urea_adjustment(sample: np.ndarray,
                            solubilization_urea: float = None,
                            urea_decrease_step: float = None,
                            dilution_increase_step: float = None,
                            max_dilution_factor: float = None,
                            max_attempts: int = None) -> np.ndarray:
    """Adjust sample parameters to satisfy urea dilution constraints.

    Uses a bounded projection strategy on final urea and dilution factor to
    reach the feasible region with minimal adjustment.

    Args:
        sample: Array containing [DTT, GSSG, dilution_factor, pH, final_urea].
        solubilization_urea: Urea concentration in solubilization buffer (M).
        urea_decrease_step: Step size for decreasing final urea concentration.
        dilution_increase_step: Step size for increasing dilution factor.
        max_dilution_factor: Maximum allowed dilution factor.
        max_attempts: Maximum number of adjustment attempts.

    Returns:
        Adjusted sample that satisfies the constraint (if possible).
    """
    # Use config defaults if not specified
    if solubilization_urea is None:
        solubilization_urea = ConstraintConfig.SOLUBILIZATION_UREA
    if urea_decrease_step is None:
        urea_decrease_step = ConstraintConfig.UREA_DECREASE_STEP
    if dilution_increase_step is None:
        dilution_increase_step = ConstraintConfig.DILUTION_INCREASE_STEP
    if max_dilution_factor is None:
        max_dilution_factor = ConstraintConfig.MAX_DILUTION_FACTOR
    if max_attempts is None:
        max_attempts = ConstraintConfig.MAX_ADJUSTMENT_ATTEMPTS

    if check_urea_constraint(sample, solubilization_urea):
        return sample

    final_urea_idx = ConstraintConfig.FINAL_UREA_IDX
    dilution_idx = ConstraintConfig.DILUTION_FACTOR_IDX
    min_dilution_factor = ConstraintConfig.MIN_DILUTION_FACTOR

    dilution_factor = sample[dilution_idx]
    final_urea = sample[final_urea_idx]

    # First try to repair by increasing final urea while keeping dilution fixed.
    required_urea = (solubilization_urea + CONSTRAINT_MARGIN) / dilution_factor
    projected_urea = min(
        ConstraintConfig.MAX_FINAL_UREA,
        max(final_urea, np.ceil(required_urea / urea_decrease_step) * urea_decrease_step),
    )
    sample[final_urea_idx] = projected_urea
    if check_urea_constraint(sample, solubilization_urea):
        return sample

    # If urea hits its bound, increase dilution to the minimum feasible value.
    required_dilution = (solubilization_urea + CONSTRAINT_MARGIN) / max(
        sample[final_urea_idx], urea_decrease_step
    )
    projected_dilution = min(
        max_dilution_factor,
        max(dilution_factor, np.ceil(required_dilution / dilution_increase_step) * dilution_increase_step),
    )
    sample[dilution_idx] = max(projected_dilution, min_dilution_factor)

    if check_urea_constraint(sample, solubilization_urea):
        return sample

    # Log warning if constraint couldn't be satisfied
    if not check_urea_constraint(sample, solubilization_urea):
        logger.warning(f"Could not satisfy urea constraint for sample: {sample}")

    return sample


def correct_constraints_iterative(samples: List[np.ndarray],
                                solubilization_urea: float = None,
                                urea_decrease_step: float = None,
                                dilution_increase_step: float = None,
                                max_dilution_factor: float = None,
                                max_attempts: int = None) -> List[np.ndarray]:
    """Apply urea dilution constraints to all samples.

    Args:
        samples: List of parameter arrays, each containing [DTT, GSSG, dilution_factor, pH, final_urea].
        solubilization_urea: Urea concentration in solubilization buffer (M).
        urea_decrease_step: Step size for decreasing final urea concentration.
        dilution_increase_step: Step size for increasing dilution factor.
        max_dilution_factor: Maximum allowed dilution factor.
        max_attempts: Maximum number of adjustment attempts per sample.

    Returns:
        List of corrected samples that satisfy the physical constraints.
    """
    # If constraint is disabled, return samples unchanged
    if not ConstraintConfig.ENABLE_UREA_CONSTRAINT:
        return samples

    corrected_samples = []
    for i, sample in enumerate(samples):
        original_sample = sample.copy()
        corrected_sample = iterative_urea_adjustment(
            sample.copy(),
            solubilization_urea,
            urea_decrease_step,
            dilution_increase_step,
            max_dilution_factor,
            max_attempts
        )
        corrected_samples.append(corrected_sample)

        # Log significant changes
        if np.any(np.abs(corrected_sample - original_sample) > 0.01):
            logger.debug(f"Sample {i} adjusted: {original_sample} -> {corrected_sample}")

    return corrected_samples


def calculate_urea_refolding_concentration(final_urea: float,
                                           dilution_factor: float,
                                           solubilization_urea: float = None) -> float:
    """Calculate the urea refolding concentration.

    Args:
        final_urea: Final urea concentration (M).
        dilution_factor: Dilution factor.
        solubilization_urea: Urea concentration in solubilization buffer (M).
                           Defaults to ConstraintConfig.SOLUBILIZATION_UREA.

    Returns:
        Urea refolding concentration (M).
    """
    if solubilization_urea is None:
        solubilization_urea = ConstraintConfig.SOLUBILIZATION_UREA

    if dilution_factor == 1:
        raise ValueError("Dilution factor cannot be 1")
    return (final_urea * dilution_factor - solubilization_urea) / (dilution_factor - 1)


def urea_constraint_callable(samples: torch.Tensor,
                            solubilization_urea: float = None,
                            bounds: torch.Tensor = None) -> torch.Tensor:
    """Constraint callable for BoTorch nonlinear constraints.

    Returns ``final_urea * dilution_factor - solubilization_urea`` so that
    feasible samples satisfy ``callable(x) > 0``.

    The function supports both a single sample of shape ``[d]`` and batched
    samples of shape ``[..., d]``. If ``bounds`` are provided, inputs are
    assumed to be normalized to ``[0, 1]`` and are denormalized internally.

    Args:
        samples: Tensor of samples ``[d]`` or ``[..., d]``.
        solubilization_urea: Urea concentration in solubilization buffer (M).
        bounds: Optional bounds tensor (2 x d).

    Returns:
        Scalar tensor for a single sample or tensor of shape ``[...]`` for
        batched samples.
    """
    if solubilization_urea is None:
        solubilization_urea = ConstraintConfig.SOLUBILIZATION_UREA

    # If bounds provided, denormalize samples
    if bounds is not None:
        samples = bounds[0] + samples * (bounds[1] - bounds[0])

    final_urea = samples[..., ConstraintConfig.FINAL_UREA_IDX]
    dilution_factor = samples[..., ConstraintConfig.DILUTION_FACTOR_IDX]

    # Return positive values for feasible samples
    return final_urea * dilution_factor - solubilization_urea


def urea_constraint_jacobian(samples: torch.Tensor,
                            solubilization_urea: float = None,
                            bounds: torch.Tensor = None) -> torch.Tensor:
    """Jacobian of the urea constraint for BoTorch.

    The constraint is: final_urea * dilution_factor > solubilization_urea
    Constraint value: f = final_urea * dilution_factor - solubilization_urea

    Partial derivatives:
        df/d(dilution_factor) = final_urea  (index 2)
        df/d(final_urea) = dilution_factor  (index 4)

    Args:
        samples: Tensor of samples [..., d] where d is the number of parameters.
        solubilization_urea: Urea concentration in solubilization buffer (M).
        bounds: Optional bounds tensor (2 x d). If provided, samples are assumed
                to be in [0,1] space and Jacobian is scaled accordingly.

    Returns:
        Tensor of Jacobian values [..., d].
    """
    if solubilization_urea is None:
        solubilization_urea = ConstraintConfig.SOLUBILIZATION_UREA

    # Get the shape for the Jacobian
    batch_shape = samples.shape[:-1]
    d = samples.shape[-1]

    # Initialize Jacobian with zeros
    jacobian = torch.zeros(*batch_shape, d, dtype=samples.dtype, device=samples.device)

    # If bounds provided, denormalize samples for value computation
    if bounds is not None:
        samples_denorm = bounds[0] + samples * (bounds[1] - bounds[0])
        # Scale factors for chain rule when computing gradient w.r.t. normalized inputs
        scale_factors = bounds[1] - bounds[0]
    else:
        samples_denorm = samples
        scale_factors = None

    final_urea = samples_denorm[..., ConstraintConfig.FINAL_UREA_IDX]
    dilution_factor = samples_denorm[..., ConstraintConfig.DILUTION_FACTOR_IDX]

    # Set partial derivatives (in original space)
    df_ddilution = final_urea
    df_durea = dilution_factor

    # If working in normalized space, apply chain rule
    if scale_factors is not None:
        df_ddilution = df_ddilution * scale_factors[ConstraintConfig.DILUTION_FACTOR_IDX]
        df_durea = df_durea * scale_factors[ConstraintConfig.FINAL_UREA_IDX]

    jacobian[..., ConstraintConfig.DILUTION_FACTOR_IDX] = df_ddilution
    jacobian[..., ConstraintConfig.FINAL_UREA_IDX] = df_durea

    return jacobian


def get_urea_constraint_tuple(solubilization_urea: float = None,
                              bounds: torch.Tensor = None) -> tuple:
    """Get a BoTorch nonlinear inequality constraint specification.

    BoTorch expects nonlinear inequality constraints as ``(callable, intra_point)``
    tuples where the callable satisfies ``callable(x) >= 0`` on feasible points.

    Args:
        solubilization_urea: Urea concentration in solubilization buffer (M).
        bounds: Bounds tensor (2 x d). If provided, the callable assumes
                normalized inputs and denormalizes internally.

    Returns:
        Tuple of ``(constraint_callable, True)`` for an intra-point constraint.
    """
    if solubilization_urea is None:
        solubilization_urea = ConstraintConfig.SOLUBILIZATION_UREA

    def constraint_fn(samples: torch.Tensor) -> torch.Tensor:
        return urea_constraint_callable(samples, solubilization_urea, bounds)

    return (constraint_fn, True)
