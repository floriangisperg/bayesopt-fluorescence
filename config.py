"""
Configuration module for Bayesian optimization of protein refolding.

Centralizes all hyperparameters, bounds, and experimental parameters
to improve maintainability and reproducibility.
"""

import numpy as np
import torch
from typing import Tuple, List, Dict, Any

# Experiment parameters
class ExperimentConfig:
    """Configuration for experimental parameters and bounds."""

    # Parameter bounds: [lower, upper]
    PARAMETER_BOUNDS: np.ndarray = np.array([
        [0, 25],     # DTT: 0-25 mM (solubilization buffer)
        [0, 2.5],    # GSSG: 0-2.5 mM
        [2, 40],     # Dilution Factor: 2-40
        [8, 11],     # pH: 8-11
        [0, 6]       # Urea in final mixture: 0-6 M
    ])

    # Parameter names (must match Excel columns)
    PARAMETER_NAMES: List[str] = [
        "DTT [mM]",
        "GSSG [mM]",
        "Dilution Factor",
        "pH",
        "Final Urea [M]"
    ]

    # Transformation of Parameter-Range (must match PARAMETER_NAMES)
    PARAMETER_TRANSFORMATION: Dict[str, Dict[str, str]] = {
        "DTT [mM]":             {"user_space": "linear",    "model_space": "linear"},
        "GSSG [mM]":            {"user_space": "linear",    "model_space": "linear"},
        "Dilution Factor":      {"user_space": "1/x",       "model_space": "1/x"},
        "pH":                   {"user_space": "linear",    "model_space": "linear"},
        "Final Urea [M]":       {"user_space": "linear",    "model_space": "linear"},
    }

    # Objective names (must match Excel columns)
    OBJECTIVE_NAMES: List[str] = [
        "Delta AEW",
        "p_proxy"
    ]

# BO optimization hyperparameters
class OptimizationConfig:
    """Configuration for Bayesian optimization parameters."""

    # Reference point for qNEHVI in REAL objective units (not standardized).
    # One value per objective, ordered like ExperimentConfig.OBJECTIVE_NAMES.
    # It is mapped to standardized model space at runtime with the fitted
    # objective scalers (see data.preprocessing.standardize_reference_point).
    REFERENCE_POINT: List[float] = [0.0, 0.0]

    # Acquisition function optimization
    BATCH_SIZE: int = 4
    MC_SAMPLES: int = 2048  # Reduced to 500 if SMOKE_TEST is set
    NUM_RESTARTS: int = 200  # Reduced to 3 if SMOKE_TEST is set
    RAW_SAMPLES: int = 2048  # Reduced to 24 if SMOKE_TEST is set

    # Optimization options
    ACQF_OPTIONS: Dict[str, Any] = {
        "batch_limit": 5,
        "maxiter": 200
    }
    SEQUENTIAL_OPTIMIZATION: bool = True

# GP model training configuration
class ModelConfig:
    """Configuration for Gaussian Process model training.

    Hyperparameters are fitted by maximizing the exact marginal
    log-likelihood to convergence (see models.gp_fitting.fit_gp_model), so
    there is no learning rate or iteration budget here.
    """

    INITIAL_NOISE_LEVEL: float = 0.05

    # Kernel parameters
    KERNEL_NU: float = 2.5  # Matérn kernel smoothness parameter (0.5, 1.5 or 2.5)

    # Model validation
    ENABLE_CROSS_VALIDATION: bool = True

# Logging and debugging
class LoggingConfig:
    """Configuration for logging and debugging."""

    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(levelname)s - %(message)s"

    # Smoke testing mode (reduces computational load for testing) is enabled by
    # setting the environment variable SMOKE_TEST; see get_optimization_params().

# Urea dilution constraint parameters
class ConstraintConfig:
    """Configuration for physical constraint handling."""

    # Enable/disable urea dilution constraint
    ENABLE_UREA_CONSTRAINT: bool = True

    # Solubilization buffer urea concentration (M)
    # This is the urea concentration in the starting solubilization buffer
    SOLUBILIZATION_UREA: float = 8.0  # M

    # Parameter indices for constraint calculation
    # Order: [DTT, GSSG, Dilution Factor, pH, Final Urea]
    DILUTION_FACTOR_IDX: int = 2
    FINAL_UREA_IDX: int = 4

# Utility functions
def get_bounds_tensor() -> torch.Tensor:
    """Get parameter bounds as torch tensor."""
    return torch.tensor(ExperimentConfig.PARAMETER_BOUNDS, dtype=torch.float64)

def get_transposed_bounds() -> torch.Tensor:
    """Get bounds in BoTorch format (2 x d tensor)."""
    lower_bounds = ExperimentConfig.PARAMETER_BOUNDS[:, 0].tolist()
    upper_bounds = ExperimentConfig.PARAMETER_BOUNDS[:, 1].tolist()
    return torch.tensor([lower_bounds, upper_bounds], dtype=torch.float64)

def get_normalized_bounds(num_features: int) -> torch.Tensor:
    """Get normalized bounds in [0, 1] range."""
    lower_bounds = torch.zeros(num_features, dtype=torch.float64)
    upper_bounds = torch.ones(num_features, dtype=torch.float64)
    return torch.stack([lower_bounds, upper_bounds])

def get_optimization_params() -> Dict[str, Any]:
    """Get optimization parameters with smoke test adjustments."""
    from os import environ

    params = {
        "mc_samples": OptimizationConfig.MC_SAMPLES,
        "num_restarts": OptimizationConfig.NUM_RESTARTS,
        "raw_samples": OptimizationConfig.RAW_SAMPLES,
    }

    if environ.get("SMOKE_TEST"):
        params.update({
            "mc_samples": 500,
            "num_restarts": 3,
            "raw_samples": 24,
        })

    return params
