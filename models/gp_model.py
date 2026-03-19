"""
Gaussian Process model definition for protein refolding optimization.

Implements a single-task GP with Matérn kernel for Bayesian optimization.
"""

import gpytorch
import botorch


class GPModel(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):
    """Single-task Gaussian Process model with Matérn kernel.

    This model combines GPyTorch's ExactGP with BoTorch's GPyTorchModel
    for compatibility with Bayesian optimization algorithms.
    """

    def __init__(self, train_x, train_y, likelihood):
        """Initialize the GP model.

        Args:
            train_x: Training inputs (normalized to [0,1]).
            train_y: Training outputs (standardized).
            likelihood: Gaussian likelihood function.
        """
        # Squeeze train_y to ensure it's 1D for GPyTorch compatibility
        train_y_squeezed = train_y.squeeze(-1)
        super().__init__(train_x, train_y_squeezed, likelihood)

        # Mean and covariance functions
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,
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