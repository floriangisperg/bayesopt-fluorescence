"""
Urea dilution constraint handling for protein refolding optimization.

Implements the physical constraint that the refolding buffer must be preparable
from the solubilization stock: ``final_urea * dilution_factor >= solubilization_urea``.
The constraint is enforced as an exact linear inequality inside the acquisition
optimizer (see ``get_urea_linear_constraint``) and validated in physical units on
exported experiment plans (see ``assert_urea_feasible``). There is deliberately
no post-hoc repair: the optimizer returns feasible candidates, and a violation
at export time indicates an upstream numerical failure that should be surfaced,
not patched.
"""

import logging
from typing import Union

import numpy as np
import torch

from config import ConstraintConfig, ExperimentConfig

logger = logging.getLogger(__name__)

# Numerical slack (M * dilution units) when validating exported plans. Orders
# of magnitude above float64 noise from the optimizer, far below experimental
# relevance.
CONSTRAINT_TOLERANCE = 1e-6


def get_urea_linear_constraint(solubilization_urea: float = None,
                               dilution_idx: int = None,
                               urea_idx: int = None) -> tuple:
    """Get the urea constraint as a BoTorch linear inequality constraint.

    The physical constraint ``final_urea * dilution_factor >= solubilization_urea``
    is exactly linear in the pipeline's unit model space, because that space
    stores the dilution factor reciprocally. With ``c = 1/dilution_factor`` the
    constraint is equivalent (for ``dilution_factor > 0``) to
    ``final_urea >= solubilization_urea * c``, and both model-space columns are
    affine functions of their unit coordinates, so the constraint reduces to a
    half-space ``coefficients . x >= rhs``.

    Returns a tuple in BoTorch's ``inequality_constraints`` format for use with
    ``optimize_acqf`` on the unit model space. BoTorch handles linear
    constraints natively: initial conditions are sampled from the feasible
    polytope and returned candidates are re-checked (and projected back onto
    the feasible set if SLSQP leaves them marginally infeasible).

    The boundary ``final_urea * dilution_factor == solubilization_urea``
    corresponds to a refolding buffer with zero urea, which is preparable and
    therefore considered feasible.

    Args:
        solubilization_urea: Urea concentration in solubilization buffer (M).
            Defaults to ConstraintConfig.SOLUBILIZATION_UREA.
        dilution_idx: Index of the dilution factor parameter. Defaults to
                      ConstraintConfig.DILUTION_FACTOR_IDX.
        urea_idx: Index of the final urea parameter. Defaults to
                  ConstraintConfig.FINAL_UREA_IDX.

    Returns:
        Tuple ``(indices, coefficients, rhs)`` such that feasibility is
        ``sum_i coefficients[i] * X[..., indices[i]] >= rhs``.
    """
    if solubilization_urea is None:
        solubilization_urea = ConstraintConfig.SOLUBILIZATION_UREA
    if dilution_idx is None:
        dilution_idx = ConstraintConfig.DILUTION_FACTOR_IDX
    if urea_idx is None:
        urea_idx = ConstraintConfig.FINAL_UREA_IDX

    # The linear form only holds under these model-space transforms; guard
    # against silently building a wrong constraint if the config changes.
    dilution_kind = ExperimentConfig.PARAMETER_TRANSFORMATION[
        ExperimentConfig.PARAMETER_NAMES[dilution_idx]
    ]["model_space"]
    urea_kind = ExperimentConfig.PARAMETER_TRANSFORMATION[
        ExperimentConfig.PARAMETER_NAMES[urea_idx]
    ]["model_space"]
    if dilution_kind not in ("1/x", "reciprocal"):
        raise ValueError(
            f"The linear urea constraint requires the dilution factor's "
            f"model-space transform to be '1/x', got {dilution_kind!r}."
        )
    if urea_kind != "linear":
        raise ValueError(
            f"The linear urea constraint requires the final urea's "
            f"model-space transform to be 'linear', got {urea_kind!r}."
        )

    d_lb, d_ub = ExperimentConfig.PARAMETER_BOUNDS[dilution_idx]
    u_lb, u_ub = ExperimentConfig.PARAMETER_BOUNDS[urea_idx]

    # Unit model space anchors x=0 at the physical lower bound, so the
    # reciprocal dilution column holds c0 = 1/d_lb at x_dil = 0 and
    # c1 = 1/d_ub at x_dil = 1. Substituting the affine unit-coordinate maps
    # into final_urea >= S * c yields the half-space below.
    c0, c1 = 1.0 / d_lb, 1.0 / d_ub
    coeff_dilution = solubilization_urea * (c0 - c1)
    coeff_urea = u_ub - u_lb
    rhs = solubilization_urea * c0 - u_lb

    return (
        torch.tensor([dilution_idx, urea_idx], dtype=torch.long),
        torch.tensor([coeff_dilution, coeff_urea], dtype=torch.float64),
        float(rhs),
    )


def assert_urea_feasible(samples: Union[np.ndarray, torch.Tensor],
                         solubilization_urea: float = None,
                         dilution_idx: int = None,
                         urea_idx: int = None,
                         tolerance: float = None) -> np.ndarray:
    """Validate physical samples against the urea dilution constraint.

    Computes ``final_urea * dilution_factor - solubilization_urea`` for each
    sample in physical units and raises ``ValueError`` if any value falls below
    ``-tolerance``. The acquisition optimizer already enforces the constraint,
    so this is a final guard on exported experiment plans: a violation here
    signals an upstream numerical failure that should be surfaced, not
    repaired.

    Args:
        samples: Parameter values in config order (default:
                [DTT, GSSG, dilution_factor, pH, final_urea]). Single sample
                ``[d]`` or batch ``[n, d]``; numpy array or torch tensor.
        solubilization_urea: Urea concentration in solubilization buffer (M).
                Defaults to ConstraintConfig.SOLUBILIZATION_UREA.
        dilution_idx: Index of the dilution factor in ``samples``. Defaults to
                      ConstraintConfig.DILUTION_FACTOR_IDX.
        urea_idx: Index of the final urea in ``samples``. Defaults to
                  ConstraintConfig.FINAL_UREA_IDX.
        tolerance: Numerical slack (M * dilution units). Defaults to
                   CONSTRAINT_TOLERANCE.

    Returns:
        Per-sample constraint values ``final_urea * dilution_factor - S``.

    Raises:
        ValueError: If any sample violates the constraint beyond ``tolerance``.
    """
    if solubilization_urea is None:
        solubilization_urea = ConstraintConfig.SOLUBILIZATION_UREA
    if dilution_idx is None:
        dilution_idx = ConstraintConfig.DILUTION_FACTOR_IDX
    if urea_idx is None:
        urea_idx = ConstraintConfig.FINAL_UREA_IDX
    if tolerance is None:
        tolerance = CONSTRAINT_TOLERANCE

    X = samples.cpu().detach().numpy() if isinstance(samples, torch.Tensor) else np.asarray(samples)
    if X.ndim == 1:
        X = X[None, :]
    if not np.isfinite(X).all():
        raise ValueError("Samples must contain only finite parameter values")

    final_urea = X[:, urea_idx]
    dilution_factor = X[:, dilution_idx]
    values = final_urea * dilution_factor - solubilization_urea

    infeasible = values < -tolerance
    if infeasible.any():
        details = []
        for i in np.where(infeasible)[0]:
            details.append(
                f"sample {i}: final_urea={final_urea[i]:.4f} M, "
                f"dilution_factor={dilution_factor[i]:.4f}, "
                f"violation={-values[i]:.3e} "
                f"(minimum feasible final_urea={solubilization_urea / dilution_factor[i]:.4f} M)"
            )
        raise ValueError(
            "Urea dilution constraint violated (final_urea * dilution_factor < "
            f"{solubilization_urea} M) in {infeasible.sum()} of {len(X)} samples:\n  "
            + "\n  ".join(details)
        )

    return values


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
    """Constraint callable for rejection sampling in physical units.

    Returns ``final_urea * dilution_factor - solubilization_urea`` so that
    feasible samples satisfy ``callable(x) >= 0``. Used by the constraint-aware
    initial design; the acquisition optimizer uses the linear form from
    ``get_urea_linear_constraint`` instead.

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
