"""Parameter transformations for experimental design and GP model inputs.

The optimization code works with two normalized spaces:

- ``model_space`` is used for GP training and acquisition optimization.
- ``user_space`` is used for experiment-facing sampling and reporting.

Both spaces are configured per parameter in ``ExperimentConfig``. The
``ParameterTransformer`` keeps the parameter order from the config and can also
transform column subsets by passing global parameter indices via ``cols``.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


def _as_array_like(value: object) -> ArrayLike:
    """Return tensors unchanged and coerce NumPy/scalar math results to float arrays."""
    if torch.is_tensor(value):
        return value
    return np.asarray(value, dtype=np.float64)


class BaseTransformation:
    """Base class for reversible one-dimensional parameter transformations.

    Subclasses implement the forward and inverse transformation for a single
    parameter. They must preserve the input backend: tensor inputs should return
    tensors, while array-like inputs should return NumPy arrays.
    """

    def __init__(self, lower_bound, upper_bound):
        """Store finite physical bounds for a parameter."""
        if not np.isfinite(lower_bound) or not np.isfinite(upper_bound):
            raise ValueError("Bounds must be finite")
        if lower_bound >= upper_bound:
            raise ValueError("Lower bound must be smaller than upper bound")
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)

    def physical_to_unit(self, x: ArrayLike) -> ArrayLike:
        """Transform values from physical units into this transform's unit space."""
        raise NotImplementedError

    def unit_to_physical(self, x: ArrayLike) -> ArrayLike:
        """Transform values from this transform's unit space back to physical units."""
        raise NotImplementedError

    def transformed_bounds(self) -> Tuple[float, float]:
        """Return the physical lower/upper bounds after forward transformation."""
        bounds = torch.tensor([self.lower_bound, self.upper_bound], dtype=torch.float64)
        transformed = self.physical_to_unit(bounds)
        if torch.is_tensor(transformed):
            transformed = transformed.detach().cpu().numpy()
        return float(transformed[0]), float(transformed[1])


class IdentityTransform(BaseTransformation):
    """Leave values in physical units."""

    def physical_to_unit(self, x: ArrayLike) -> ArrayLike:
        return _as_array_like(x)

    def unit_to_physical(self, x: ArrayLike) -> ArrayLike:
        return _as_array_like(x)


class LinearScaler(BaseTransformation):
    """Scale physical values linearly to and from the interval [0, 1]."""

    def physical_to_unit(self, x: ArrayLike) -> ArrayLike:
        return _as_array_like(
            (x - self.lower_bound) / (self.upper_bound - self.lower_bound)
        )

    def unit_to_physical(self, x: ArrayLike) -> ArrayLike:
        return _as_array_like(
            x * (self.upper_bound - self.lower_bound) + self.lower_bound
        )


class ReciprocalScaler(BaseTransformation):
    """Scale the reciprocal of a non-zero physical value to and from [0, 1].

    This is useful when equal spacing in reciprocal space is more meaningful
    than equal spacing in the raw physical value, for example dilution factors.
    """

    def __init__(self, lower_bound, upper_bound):
        super().__init__(lower_bound, upper_bound)
        if self.lower_bound == 0 or self.upper_bound == 0:
            raise ValueError("Bound for 1/x-transformation must not be zero")
        if self.lower_bound < 0 < self.upper_bound:
            raise ValueError("Bounds for 1/x-transformation must not cross zero")

    def physical_to_unit(self, x: ArrayLike) -> ArrayLike:
        y = 1 / x
        y_lb = 1 / self.lower_bound
        y_ub = 1 / self.upper_bound
        transformed = (y - y_lb) / (y_ub - y_lb)
        return _as_array_like(transformed)

    def unit_to_physical(self, x: ArrayLike) -> ArrayLike:
        y_lb = 1 / self.lower_bound
        y_ub = 1 / self.upper_bound
        y = x * (y_ub - y_lb) + y_lb
        transformed = 1 / y
        return _as_array_like(transformed)


class LogScaler(BaseTransformation):
    """Scale the natural logarithm of a positive physical value to [0, 1].

    Use this for strictly positive parameters where multiplicative changes are
    more meaningful than additive changes.
    """

    def __init__(self, lower_bound, upper_bound):
        super().__init__(lower_bound, upper_bound)
        if self.lower_bound <= 0:
            raise ValueError("Bounds for log-transformation must be positive")
        self._log_lower = np.log(self.lower_bound)
        self._log_upper = np.log(self.upper_bound)

    def physical_to_unit(self, x: ArrayLike) -> ArrayLike:
        y = torch.log(x) if torch.is_tensor(x) else np.log(x)
        return _as_array_like(
            (y - self._log_lower) / (self._log_upper - self._log_lower)
        )

    def unit_to_physical(self, x: ArrayLike) -> ArrayLike:
        y = x * (self._log_upper - self._log_lower) + self._log_lower
        return _as_array_like(torch.exp(y) if torch.is_tensor(y) else np.exp(y))


class LogitScaler(BaseTransformation):
    """Scale a bounded physical value through a logit transform.

    The physical bounds define the open interval mapped by the logit. Inputs at
    exact bounds are clipped by ``eps`` to avoid infinities, so the transform is
    numerically stable for boundary values.
    """

    eps = 1e-12

    def __init__(self, lower_bound, upper_bound):
        super().__init__(lower_bound, upper_bound)
        z_bounds = np.array([self.eps, 1.0 - self.eps], dtype=np.float64)
        logits = np.log(z_bounds / (1.0 - z_bounds))
        self._logit_lower = float(logits[0])
        self._logit_upper = float(logits[1])

    def _to_interval(self, x: ArrayLike) -> ArrayLike:
        z = (x - self.lower_bound) / (self.upper_bound - self.lower_bound)
        if torch.is_tensor(z):
            return z.clamp(self.eps, 1.0 - self.eps)
        return np.asarray(np.clip(z, self.eps, 1.0 - self.eps), dtype=np.float64)

    def physical_to_unit(self, x: ArrayLike) -> ArrayLike:
        z = self._to_interval(x)
        logit = torch.logit(z) if torch.is_tensor(z) else np.log(z / (1.0 - z))
        return _as_array_like(
            (logit - self._logit_lower) / (self._logit_upper - self._logit_lower)
        )

    def unit_to_physical(self, x: ArrayLike) -> ArrayLike:
        logit = x * (self._logit_upper - self._logit_lower) + self._logit_lower
        z = (
            torch.sigmoid(logit)
            if torch.is_tensor(logit)
            else 1.0 / (1.0 + np.exp(-logit))
        )
        return _as_array_like(z * (self.upper_bound - self.lower_bound) + self.lower_bound)


TRANSFORM_REGISTRY = {
    "linear": LinearScaler,
    "1/x": ReciprocalScaler,
    "reciprocal": ReciprocalScaler,
    "log": LogScaler,
    "logit": LogitScaler,
}


@dataclass(frozen=True)
class ParameterSpec:
    """Fully resolved transformation metadata for one configured parameter."""

    name: str
    model_transform: BaseTransformation
    user_transform: BaseTransformation
    lower_bound: float
    upper_bound: float


class ParameterTransformer:
    """Transform configured experiment parameters between physical and unit spaces.

    ``model_space`` transformations are used for GP training and acquisition
    optimization. ``user_space`` transformations are used for experiment-facing
    sampling and reporting. Column subsets can be transformed by passing the
    global parameter indices via ``cols``.

    Shape rules:
    - Inputs with two or more dimensions use the final dimension as
      ``n_selected_parameters`` and preserve all leading batch dimensions.
    - 1D inputs with one selected column are interpreted as many values for that
      parameter.
    - 1D inputs with multiple selected columns are interpreted as one full row.
    """

    def __init__(
            self,
            parameter_names: Sequence[str],
            parameter_bounds: np.ndarray,
            parameter_transforms: Dict[str, Union[str, dict]],
    ):

        self.parameter_names = list(parameter_names)
        self.parameter_bounds = np.asarray(parameter_bounds, dtype=np.float64)
        self.parameter_transforms = parameter_transforms

        if set(self.parameter_names) != set(self.parameter_transforms.keys()):
            raise ValueError("Items in PARAMETER_TRANSFORMATION must match PARAMETER_NAMES (config.py)")

        if len(self.parameter_names) != len(self.parameter_bounds):
            raise ValueError(
                "Number of elements in PARAMETER_NAMES must match number of rows "
                "in PARAMETER_BOUNDS (config.py)"
            )

        self.specs: List[ParameterSpec] = []
        for i, name in enumerate(self.parameter_names):
            lower_bound, upper_bound = self.parameter_bounds[i]

            model_transform_kind = self.parameter_transforms[name]["model_space"]
            model_transform_cls = TRANSFORM_REGISTRY.get(model_transform_kind)

            user_transform_kind = self.parameter_transforms[name]["user_space"]
            user_transform_cls = TRANSFORM_REGISTRY.get(user_transform_kind)

            if model_transform_cls is None:
                raise ValueError(
                    f"Unknown model_space transform kind {model_transform_kind!r} for parameter {name}"
                )
            if user_transform_cls is None:
                raise ValueError(
                    f"Unknown user_space transform kind {user_transform_kind!r} for parameter {name}"
                )

            model_transform = model_transform_cls(lower_bound, upper_bound)
            user_transform = user_transform_cls(lower_bound, upper_bound)

            self.specs.append(
                ParameterSpec(
                    name=name,
                    model_transform=model_transform,
                    user_transform=user_transform,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                )
            )

        self._lower_bound = self._vector("lower_bound")
        self._upper_bound = self._vector("upper_bound")

    def _vector(self, attr: str) -> np.ndarray:
        """Collect one scalar attribute from all parameter specs in config order."""
        return np.array([getattr(spec, attr) for spec in self.specs], dtype=float)

    @staticmethod
    def _as_2d(x: ArrayLike, cols: List[int]) -> tuple[torch.Tensor, Tuple[int, ...]]:
        """Convert input to a flat float64 matrix and retain its original shape."""
        if torch.is_tensor(x):
            xt = x.double()
        else:
            xt = torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=torch.float64)

        n_cols = len(cols)
        original_shape = tuple(xt.shape)
        if xt.ndim == 0:
            if n_cols != 1:
                raise ValueError("Scalar inputs can only be transformed with exactly one column.")
            return xt.reshape(1, 1), original_shape

        if xt.ndim == 1:
            if n_cols == 1:
                return xt.reshape(-1, 1), original_shape
            if xt.shape[0] == n_cols:
                return xt.reshape(1, n_cols), original_shape
            raise ValueError(
                f"One-dimensional input has length {xt.shape[0]}, but {n_cols} columns were selected. "
                "Pass a 2D array for multiple samples or pass cols=[...] for a single parameter."
            )
        if xt.shape[-1] != n_cols:
            raise ValueError(
                f"Expected final dimension with {n_cols} columns, got {xt.shape[-1]}."
            )
        return xt.reshape(-1, n_cols), original_shape

    @staticmethod
    def _restore_shape(x: torch.Tensor, original_shape: Tuple[int, ...], as_tensor: bool) -> ArrayLike:
        """Return transformed data with the caller's original shape convention."""
        x = x.reshape(original_shape)
        if as_tensor:
            return x
        return x.detach().cpu().numpy()

    @staticmethod
    def _to_tensor(x: ArrayLike) -> torch.Tensor:
        """Normalize transform outputs before assigning them into tensor columns."""
        if torch.is_tensor(x):
            return x.double()
        return torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=torch.float64)

    def _resolve_cols(self, cols: Optional[Sequence[int]]) -> List[int]:
        """Return validated global parameter indices for the selected columns."""
        if cols is None:
            return list(range(len(self.specs)))
        resolved = list(cols)
        invalid_cols = [col for col in resolved if col < 0 or col >= len(self.specs)]
        if invalid_cols:
            raise IndexError(f"Column indices out of range: {invalid_cols}")
        if not resolved:
            raise ValueError("At least one column must be selected")
        return resolved

    def physical_to_unit_model(
            self,
            x: ArrayLike,
            cols: Sequence[int] = None,
            as_tensor: bool = False,
    ) -> ArrayLike:
        """Transform physical values to model-space unit values."""
        cols = self._resolve_cols(cols)
        x2d, original_shape = self._as_2d(x, cols)
        z = x2d.clone()
        for local_i, global_i in enumerate(cols):
            spec = self.specs[global_i]
            # ``local_i`` addresses the provided matrix; ``global_i`` addresses config order.
            z[:, local_i] = self._to_tensor(
                spec.model_transform.physical_to_unit(x2d[:, local_i])
            )
        return self._restore_shape(z, original_shape, as_tensor)

    def unit_to_physical_model(
            self,
            z: ArrayLike,
            cols: Sequence[int] = None,
            as_tensor: bool = False,
    ) -> ArrayLike:
        """Transform model-space unit values back to physical values."""
        cols = self._resolve_cols(cols)
        z2d, original_shape = self._as_2d(z, cols)
        x = z2d.clone()
        for local_i, global_i in enumerate(cols):
            spec = self.specs[global_i]
            x[:, local_i] = self._to_tensor(
                spec.model_transform.unit_to_physical(z2d[:, local_i])
            )
        return self._restore_shape(x, original_shape, as_tensor)

    def physical_to_unit_user(
            self,
            x: ArrayLike,
            cols: Sequence[int] = None,
            as_tensor: bool = False,
    ) -> ArrayLike:
        """Transform physical values to user-space unit values."""
        cols = self._resolve_cols(cols)
        x2d, original_shape = self._as_2d(x, cols)
        z = x2d.clone()
        for local_i, global_i in enumerate(cols):
            spec = self.specs[global_i]
            z[:, local_i] = self._to_tensor(
                spec.user_transform.physical_to_unit(x2d[:, local_i])
            )
        return self._restore_shape(z, original_shape, as_tensor)

    def unit_to_physical_user(
            self,
            z: ArrayLike,
            cols: Sequence[int] = None,
            as_tensor: bool = False,
    ) -> ArrayLike:
        """Transform user-space unit values back to physical values."""
        cols = self._resolve_cols(cols)
        z2d, original_shape = self._as_2d(z, cols)
        x = z2d.clone()
        for local_i, global_i in enumerate(cols):
            spec = self.specs[global_i]
            x[:, local_i] = self._to_tensor(
                spec.user_transform.unit_to_physical(z2d[:, local_i])
            )
        return self._restore_shape(x, original_shape, as_tensor)

    def get_physical_bounds(self, as_tensor: bool = False) -> ArrayLike:
        """Return physical parameter bounds as a ``2 x d`` array or tensor."""
        bounds = np.stack([self._lower_bound, self._upper_bound], axis=0)
        return torch.tensor(bounds, dtype=torch.float64) if as_tensor else bounds

    def get_model_bounds(self, as_tensor: bool = False) -> ArrayLike:
        """Return bounds transformed with each parameter's model-space transform."""
        bounds = np.array([spec.model_transform.transformed_bounds() for spec in self.specs], dtype=float).T
        return torch.tensor(bounds, dtype=torch.float64) if as_tensor else bounds

    def get_user_bounds(self, as_tensor: bool = False) -> ArrayLike:
        """Return bounds transformed with each parameter's user-space transform."""
        bounds = np.array([spec.user_transform.transformed_bounds() for spec in self.specs], dtype=float).T
        return torch.tensor(bounds, dtype=torch.float64) if as_tensor else bounds


def build_transformer(config) -> ParameterTransformer:
    """Build a parameter transformer from a config class or config-like object."""
    return ParameterTransformer(
        parameter_names=config.PARAMETER_NAMES,
        parameter_bounds=config.PARAMETER_BOUNDS,
        parameter_transforms=config.PARAMETER_TRANSFORMATION,
    )
