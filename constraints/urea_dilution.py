"""
Urea dilution constraint handling for protein refolding optimization.

Implements physical constraints for the urea dilution process to ensure
that generated experimental conditions are physically feasible.
"""

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


def check_urea_constraint(sample: np.ndarray) -> bool:
    """Check if a sample satisfies the urea dilution physical constraint.

    The constraint ensures that the refolding urea concentration is positive:
    urea_refolding = (final_urea * dilution_factor - 8) / (dilution_factor - 1) > 0

    Args:
        sample: Array containing [DTT, GSSG, dilution_factor, pH, final_urea].

    Returns:
        True if the constraint is satisfied, False otherwise.
    """
    final_urea, dilution_factor = sample[4], sample[2]
    urea_refolding = ((final_urea * dilution_factor) - 8) / (dilution_factor - 1)
    return urea_refolding > 0


def iterative_urea_adjustment(sample: np.ndarray,
                            urea_decrease_step: float = 0.1,
                            dilution_increase_step: float = 0.5,
                            max_dilution_factor: float = 40,
                            max_attempts: int = 500) -> np.ndarray:
    """Adjust sample parameters to satisfy urea dilution constraints.

    Uses an alternating adjustment strategy to modify final urea concentration
    and dilution factor until the physical constraint is satisfied.

    Args:
        sample: Array containing [DTT, GSSG, dilution_factor, pH, final_urea].
        urea_decrease_step: Step size for decreasing final urea concentration.
        dilution_increase_step: Step size for increasing dilution factor.
        max_dilution_factor: Maximum allowed dilution factor.
        max_attempts: Maximum number of adjustment attempts.

    Returns:
        Adjusted sample that satisfies the constraint (if possible).
    """
    adjustment_count = 0

    while adjustment_count < max_attempts:
        if check_urea_constraint(sample):
            return sample

        # Alternating adjustment strategy
        if adjustment_count % 2 == 0:
            # Adjust final urea concentration
            if sample[4] - urea_decrease_step >= 0:
                sample[4] -= urea_decrease_step
            else:
                adjustment_count += 1
                continue
        else:
            # Adjust dilution factor
            if (sample[2] + dilution_increase_step <= max_dilution_factor):
                sample[2] += dilution_increase_step
            else:
                adjustment_count += 1
                continue

        adjustment_count += 1

    # Log warning if constraint couldn't be satisfied
    if not check_urea_constraint(sample):
        logger.warning(f"Could not satisfy urea constraint for sample: {sample}")

    return sample


def correct_constraints_iterative(samples: List[np.ndarray],
                                urea_decrease_step: float = 0.1,
                                dilution_increase_step: float = 0.5,
                                max_dilution_factor: float = 40,
                                max_attempts: int = 500) -> List[np.ndarray]:
    """Apply urea dilution constraints to all samples.

    Args:
        samples: List of parameter arrays, each containing [DTT, GSSG, dilution_factor, pH, final_urea].
        urea_decrease_step: Step size for decreasing final urea concentration.
        dilution_increase_step: Step size for increasing dilution factor.
        max_dilution_factor: Maximum allowed dilution factor.
        max_attempts: Maximum number of adjustment attempts per sample.

    Returns:
        List of corrected samples that satisfy the physical constraints.
    """
    corrected_samples = []
    for i, sample in enumerate(samples):
        original_sample = sample.copy()
        corrected_sample = iterative_urea_adjustment(
            sample.copy(),
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


def calculate_urea_refolding_concentration(final_urea: float, dilution_factor: float) -> float:
    """Calculate the urea refolding concentration.

    Args:
        final_urea: Final urea concentration (M).
        dilution_factor: Dilution factor.

    Returns:
        Urea refolding concentration (M).
    """
    if dilution_factor == 1:
        raise ValueError("Dilution factor cannot be 1")
    return (final_urea * dilution_factor - 8) / (dilution_factor - 1)