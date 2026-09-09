"""
qNEHVI (Noisy Expected Hypervolume Improvement) acquisition function.

Implements the qNEHVI acquisition function for multi-objective Bayesian
optimization in normalized design space and supports linear input-space
constraints for physical feasibility.
"""

import logging
import time
from typing import List, Optional, Tuple

import torch
from botorch.acquisition.multi_objective import (
    IdentityMCMultiOutputObjective,
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.optim import optimize_acqf

from config import OptimizationConfig

logger = logging.getLogger(__name__)


def create_qnehvi_acquisition(model,
                            reference_point: torch.Tensor,
                            sampler: object,
                            X_baseline: Optional[torch.Tensor] = None,
                            n_objectives: int = 2) -> qLogNoisyExpectedHypervolumeImprovement:
    """Create a qNEHVI acquisition function.

    Args:
        model: Multi-output GP model (ModelListGP).
        reference_point: Reference point for hypervolume calculation.
        sampler: MC sampler for acquisition function (e.g., SobolQMCNormalSampler).
        X_baseline: Baseline observations (optional).
        n_objectives: Number of objectives (default: 2).

    Returns:
        Configured qNEHVI acquisition function.
    """
    # Standard objective for multi-output optimization
    objective = IdentityMCMultiOutputObjective(outcomes=list(range(n_objectives)))

    return qLogNoisyExpectedHypervolumeImprovement(
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
                    sequential: bool = True,
                    inequality_constraints: Optional[List[Tuple[torch.Tensor, torch.Tensor, float]]] = None,
                    return_metadata: bool = False):
    """Optimize the qNEHVI acquisition function.

    Args:
        acq_function: Acquisition function to optimize.
        bounds: Bounds for optimization (2 x d tensor, unit model space).
        batch_size: Number of candidates to generate.
        mc_samples: Number of Monte Carlo samples.
        num_restarts: Number of optimization restarts.
        raw_samples: Number of raw samples for initialization.
        sequential: Whether to use sequential optimization.
        inequality_constraints: Optional list of BoTorch linear constraint tuples
                        ``(indices, coefficients, rhs)``. Initial conditions are
                        sampled from the feasible polytope and candidates are
                        verified (and projected) by BoTorch.
        return_metadata: Also return a metadata dict (optimizer settings,
                        runtime, final acquisition value) for run reporting.

    Returns:
        Optimized candidate points (batch_size x d), or a tuple
        ``(candidates, metadata)`` when ``return_metadata`` is True.
    """
    logger.info(f"Optimizing qNEHVI with batch_size={batch_size}, mc_samples={mc_samples}")

    started_at = time.perf_counter()
    metadata = {
        "batch_size": batch_size,
        "mc_samples": mc_samples,
        "num_restarts": num_restarts,
        "raw_samples": raw_samples,
        "sequential": sequential,
        "linear_constraints": len(inequality_constraints or []),
    }

    if inequality_constraints:
        logger.info(f"Using {len(inequality_constraints)} linear constraint(s)")

    batch_limit = OptimizationConfig.ACQF_OPTIONS.get("batch_limit", 5)
    maxiter = OptimizationConfig.ACQF_OPTIONS.get("maxiter", 200)
    metadata["options"] = {"batch_limit": batch_limit, "maxiter": maxiter}

    candidates, acq_value = optimize_acqf(
        acq_function=acq_function,
        bounds=bounds,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        sequential=sequential,
        options={"batch_limit": batch_limit, "maxiter": maxiter},
        inequality_constraints=inequality_constraints
    )

    metadata["runtime_seconds"] = time.perf_counter() - started_at
    if torch.is_tensor(acq_value):
        metadata["final_acquisition_value"] = acq_value.detach().cpu().reshape(-1).tolist()
    else:
        metadata["final_acquisition_value"] = acq_value

    logger.info(f"Generated {candidates.shape[0]} candidate points")
    if return_metadata:
        return candidates, metadata
    return candidates
