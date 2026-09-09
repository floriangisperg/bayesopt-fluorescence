"""
Gaussian Process model definition for protein refolding optimization.

Implements a single-task GP with Matérn kernel for Bayesian optimization.
"""

import botorch
import gpytorch

from config import ModelConfig


class GPModel(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):
    """Single-task Gaussian Process model with Matérn kernel.

    This model combines GPyTorch's ExactGP with BoTorch's GPyTorchModel
    for compatibility with Bayesian optimization algorithms.
    """

    def __init__(self, train_x, train_y, likelihood):
        """Initialize the GP model.

        Args:
            train_x: Training inputs (normalized to [0,1]).
            train_y: Training outputs (standardized), 1D or a single-column 2D
                     tensor (one objective only).
            likelihood: Gaussian likelihood function.
        """
        if train_y.ndim == 2 and train_y.shape[1] == 1:
            train_y = train_y.squeeze(-1)
        if train_y.ndim != 1:
            raise ValueError(
                "GPModel is single-task and expects 1D targets (or a "
                f"single-column 2D tensor), got shape {tuple(train_y.shape)}. "
                "Slice one objective column before constructing the model."
            )

        # Initialize ExactGP with 1D targets (GPyTorch requirement)
        super().__init__(train_x, train_y, likelihood)

        # Store num_outputs for BoTorch compatibility
        self._num_outputs = 1

        # Mean and covariance functions
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=ModelConfig.KERNEL_NU,
                ard_num_dims=train_x.shape[1]  # Automatic Relevance Determination
            )
        )

    def forward(self, x):
        """Forward pass through the model.

        Args:
            x: Input tensor.

        Returns:
            Multivariate normal distribution.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    @property
    def num_outputs(self) -> int:
        """Number of outputs from the model."""
        return self._num_outputs

    # Override load_state_dict to avoid BoTorch's target extraction
    def load_state_dict(self, state_dict, strict=True):
        """Load state dict, bypassing BoTorch's target extraction."""
        # Call GPyTorch's load_state_dict directly to avoid BoTorch's validation
        return gpytorch.models.ExactGP.load_state_dict(self, state_dict, strict=strict)
