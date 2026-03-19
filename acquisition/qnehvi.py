"""
qNEHVI (Noisy Expected Hypervolume Improvement) acquisition function.

Implements the qNEHVI acquisition function for multi-objective Bayesian optimization.
"""

import logging
from typing import Optional

import torch
from botorch.acquisition.multi_objective import qNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective import IdentityMCMultiOutputObjective
from botorch.optim import optimize_acqf

logger = logging.getLogger(__name__)


def create_qnehvi_acquisition(model,
                            reference_point: torch.Tensor,
                            sampler: object,
                            X_baseline: Optional[torch.Tensor] = None) -> qNoisyExpectedHypervolumeImprovement:
    """Create a qNEHVI acquisition function.

    Args:
        model: Multi-output GP model (ModelListGP).
        reference_point: Reference point for hypervolume calculation.
        sampler: MC sampler for acquisition function (e.g., SobolQMCNormalSampler).
        X_baseline: Baseline observations (optional).

    Returns:
        Configured qNEHVI acquisition function.
    """
    # Standard objective for multi-output optimization
    objective = IdentityMCMultiOutputObjective(outcomes=[0, 1])

    return qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=reference_point,
        X_baseline=X_baseline,
        sampler=sampler,
        prune_baseline=True,
        objective=objective
    )


def optimize_qnehvi(acq_function, bounds: torch.Tensor,
                    batch_size: int = 4, mc_samples: int = 2048,
                    num_restarts: int = 200, raw_samples: int = 2048,
                    sequential: bool = True) -> torch.Tensor:
    """Optimize the qNEHVI acquisition function.

    Args:
        acq_function: Acquisition function to optimize.
        bounds: Bounds for optimization (2 x d tensor).
        batch_size: Number of candidates to generate.
        mc_samples: Number of Monte Carlo samples.
        num_restarts: Number of optimization restarts.
        raw_samples: Number of raw samples for initialization.
        sequential: Whether to use sequential optimization.

    Returns:
        Optimized candidate points (batch_size x d).
    """
    logger.info(f"Optimizing qNEHVI with batch_size={batch_size}, mc_samples={mc_samples}")

    candidates, _ = optimize_acqf(
        acq_function=acq_function,
        bounds=bounds,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        sequential=sequential,
        options={"batch_limit": 5, "maxiter": 200}
    )

    logger.info(f"Generated {candidates.shape[0]} candidate points")
    return candidates