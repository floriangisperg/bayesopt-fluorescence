"""
qNEHVI (Noisy Expected Hypervolume Improvement) acquisition function.

Implements the qNEHVI acquisition function for multi-objective Bayesian
optimization in normalized design space and supports nonlinear constraints for
physical feasibility.
"""

import logging
from typing import Optional, List, Callable

import torch
from botorch.acquisition.multi_objective import qLogNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective import IdentityMCMultiOutputObjective
from botorch.optim import optimize_acqf
from botorch.optim.parameter_constraints import evaluate_feasibility
from constraints.urea_dilution import get_urea_constraint_tuple

logger = logging.getLogger(__name__)


def gen_feasible_initial_conditions(
    acq_function,
    bounds: torch.Tensor,
    q: int,
    num_restarts: int,
    raw_samples: int,
    nonlinear_inequality_constraints: List[tuple[Callable, bool]],
    **kwargs
) -> torch.Tensor:
    """Generate initial conditions that satisfy nonlinear constraints.

    Uses rejection sampling to find feasible starting points for optimization.

    Args:
        acq_function: Acquisition function.
        bounds: Bounds for optimization (2 x d tensor).
        q: Number of candidates.
        num_restarts: Number of optimization restarts.
        raw_samples: Number of raw samples for rejection sampling.
        nonlinear_inequality_constraints: List of BoTorch nonlinear constraint
            specifications ``(constraint_fn, intra_point)``.
        **kwargs: Additional arguments (ignored).

    Returns:
        Initial conditions tensor (num_restarts x q x d).
    """
    d = bounds.shape[1]
    n_required = num_restarts * q
    oversample_factor = 10
    max_sampling_rounds = 5
    feasible_batches = []
    total_drawn = 0

    for constraint_fn, intra_point in nonlinear_inequality_constraints:
        if not intra_point:
            raise NotImplementedError("Only intra-point nonlinear constraints are supported.")

    for _ in range(max_sampling_rounds):
        n_samples_needed = max(raw_samples, n_required * oversample_factor)
        samples = torch.rand(n_samples_needed, d, dtype=bounds.dtype, device=bounds.device)
        feasible_mask = evaluate_feasibility(
            X=samples.unsqueeze(-2),
            nonlinear_inequality_constraints=nonlinear_inequality_constraints,
        )
        feasible_samples = samples[feasible_mask]
        feasible_batches.append(feasible_samples)
        total_drawn += n_samples_needed
        if sum(batch.shape[0] for batch in feasible_batches) >= n_required:
            break

    feasible_samples = torch.cat(feasible_batches, dim=0) if feasible_batches else torch.empty(
        0, d, dtype=bounds.dtype, device=bounds.device
    )
    n_feasible = feasible_samples.shape[0]
    logger.info(f"Found {n_feasible} feasible samples out of {total_drawn}")

    if n_feasible == 0:
        raise RuntimeError(
            "Could not find any feasible initial conditions! "
            "The constraint may be too restrictive or bounds may be incorrect."
        )

    if n_feasible < n_required:
        logger.warning(
            f"Only found {n_feasible} feasible samples after {max_sampling_rounds} rounds. "
            f"Needed {n_required}. Reusing the best feasible starts."
        )

    with torch.no_grad():
        acq_values = acq_function(feasible_samples.unsqueeze(-2)).reshape(-1)

    n_to_select = min(n_required, n_feasible)
    _, top_indices = torch.topk(acq_values, n_to_select)
    top_samples = feasible_samples[top_indices]

    if top_samples.shape[0] < n_required:
        repeat_factor = (n_required + top_samples.shape[0] - 1) // top_samples.shape[0]
        top_samples = top_samples.repeat(repeat_factor, 1)

    return top_samples[:n_required].reshape(num_restarts, q, d)


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
                    nonlinear_inequality_constraints: Optional[List[tuple[Callable, bool]]] = None) -> torch.Tensor:
    """Optimize the qNEHVI acquisition function.

    Args:
        acq_function: Acquisition function to optimize.
        bounds: Bounds for optimization (2 x d tensor).
        batch_size: Number of candidates to generate.
        mc_samples: Number of Monte Carlo samples.
        num_restarts: Number of optimization restarts.
        raw_samples: Number of raw samples for initialization.
        sequential: Whether to use sequential optimization.
        nonlinear_inequality_constraints: Optional list of BoTorch nonlinear
                              constraint tuples ``(callable, intra_point)``.

    Returns:
        Optimized candidate points (batch_size x d).
    """
    logger.info(f"Optimizing qNEHVI with batch_size={batch_size}, mc_samples={mc_samples}")

    # Set up ic_generator for nonlinear constraints
    ic_generator = None
    if nonlinear_inequality_constraints:
        logger.info(f"Using {len(nonlinear_inequality_constraints)} nonlinear constraint(s)")
        def ic_gen(acq_function, bounds, q, num_restarts, raw_samples, **kwargs):
            return gen_feasible_initial_conditions(
                acq_function=acq_function,
                bounds=bounds,
                q=q,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                nonlinear_inequality_constraints=nonlinear_inequality_constraints,
                **kwargs
            )
        ic_generator = ic_gen

    batch_limit = 1 if nonlinear_inequality_constraints else 5

    candidates, _ = optimize_acqf(
        acq_function=acq_function,
        bounds=bounds,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        sequential=sequential,
        options={"batch_limit": batch_limit, "maxiter": 200},
        nonlinear_inequality_constraints=nonlinear_inequality_constraints,
        ic_generator=ic_generator
    )

    logger.info(f"Generated {candidates.shape[0]} candidate points")
    return candidates


def get_urea_constraint_callable(solubilization_urea: float = None,
                                  bounds: torch.Tensor = None) -> tuple:
    """Get a urea constraint tuple for BoTorch's nonlinear_inequality_constraints.

    Args:
        solubilization_urea: Urea concentration in solubilization buffer (M).
        bounds: Bounds tensor (2 x d) of the original design space. If provided,
            the constraint callable assumes normalized inputs and denormalizes
            internally.

    Returns:
        Tuple of ``(constraint_callable, True)`` for use with ``optimize_acqf``.
    """
    if bounds is None:
        from config import ExperimentConfig
        bounds = torch.from_numpy(ExperimentConfig.PARAMETER_BOUNDS.T).double()

    return get_urea_constraint_tuple(solubilization_urea, bounds)
