# TODO: add more transformations, e.g. log, logit, etc. and support for categorical parameters (e.g. one-hot encoding)
# TODO: documentation
# TODO: bound handeling, transform bounds as tensor
from dataclasses import dataclass
from typing import Dict, List, Sequence, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]

class BaseTransformation:
    def __init__(self, lower_bound, upper_bound):
        if lower_bound > upper_bound:
            raise ValueError("Lower bound must be smaller than upper bound")
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def physical_to_unit(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def unit_to_physical(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError



class IdentityTransform(BaseTransformation):

    def physical_to_unit(self, x: ArrayLike) -> ArrayLike:
        return x

    def unit_to_physical(self, x: ArrayLike) -> ArrayLike:
        return x

class LinearScaler(BaseTransformation):

    def physical_to_unit(self, x: ArrayLike) -> ArrayLike:
        return (x - self.lower_bound) / (self.upper_bound - self.lower_bound)

    def unit_to_physical(self, x: ArrayLike) -> ArrayLike:
        return x * (self.upper_bound - self.lower_bound) + self.lower_bound


class ReciprocalScaler(BaseTransformation):
    def __init__(self, lower_bound, upper_bound):
        super().__init__(lower_bound, upper_bound)
        if self.lower_bound == 0 or self.upper_bound == 0:
            raise ValueError("Bound for 1/x-transformation must not be zero")


    def physical_to_unit(self, x: ArrayLike) -> ArrayLike:
        y = 1 / x
        y_lb =  1 / self.lower_bound
        y_ub = 1 / self.upper_bound
        return (y - y_lb) / (y_ub- y_lb)

    def unit_to_physical(self, x: ArrayLike) -> ArrayLike:
        y_lb = 1 / self.lower_bound
        y_ub = 1 / self.upper_bound
        y = x * (y_ub - y_lb) + y_lb
        return 1/y


TRANSFORM_REGISTRY = {
    "none": IdentityTransform,
    "linear": LinearScaler,
    "1/x": ReciprocalScaler,
}


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    model_transform: BaseTransformation
    user_transform: BaseTransformation
    lower_bound: float
    upper_bound: float


class ParameterTransformer:
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
            raise ValueError("Number of elements in PARAMETER_NAMES must match number of rows in PARAMETER_BOUNDS (config.py)")

        self.specs: List[ParameterSpec] = []
        for i, name in enumerate(self.parameter_names):
            lower_bound, upper_bound = self.parameter_bounds[i]

            model_transform_kind = self.parameter_transforms[name]["model_space"]
            model_transform_cls = TRANSFORM_REGISTRY.get(model_transform_kind)

            user_transform_kind = self.parameter_transforms[name]["user_space"]
            user_transform_cls = TRANSFORM_REGISTRY.get(user_transform_kind)

            if model_transform_cls is None or user_transform_cls is None:
                raise ValueError(f"Unknown transform kind {model_transform_cls} for parameter {name}")

            model_transform = model_transform_cls(lower_bound, upper_bound) # TODO: check if bounds need to be transformed
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
        return np.array([getattr(spec, attr) for spec in self.specs], dtype=float)

    @staticmethod
    def _as_2d(x: ArrayLike) -> tuple[ArrayLike, bool]:
        if torch.is_tensor(x):
            if x.ndim == 1:
                return x.unsqueeze(1), True
            if x.ndim == 2:
                return x, False
        else:
            x = np.asarray(x, dtype=float)
            if x.ndim == 1:
                return x[:, None], True
            if x.ndim == 2:
                return x, False
        raise ValueError("Expected shape (n,) for one parameter or (n, d) for multiple parameters.")

    @staticmethod
    def _restore_shape(x: ArrayLike, squeeze: bool) -> ArrayLike:
        return x[:, 0] if squeeze else x

    def _resolve_cols(self, cols: List[int]) -> List[int]:
        if cols is None:
            return list(range(len(self.specs)))
        else:
            return cols

    def physical_to_unit_model(self, x: ArrayLike, cols: List = None) -> ArrayLike:
        x2d, squeeze = self._as_2d(x)
        cols = self._resolve_cols(cols)
        z = x2d.clone() if torch.is_tensor(x2d) else x2d.copy()
        for local_i, global_i in enumerate(cols):
            spec = self.specs[global_i]
            z[:, local_i] = spec.model_transform.physical_to_unit(x2d[:, local_i])
        return self._restore_shape(z, squeeze)

    def unit_to_physical_model(self, z: ArrayLike, cols: List = None) -> ArrayLike:
        z2d, squeeze = self._as_2d(z)
        cols = self._resolve_cols(cols)
        x = z2d.clone() if torch.is_tensor(z2d) else z2d.copy()
        for local_i, global_i in enumerate(cols):
            spec = self.specs[global_i]
            x[:, local_i] = spec.model_transform.unit_to_physical(z2d[:, local_i])
        return self._restore_shape(x, squeeze)

    def physical_to_unit_user(self, x: ArrayLike, cols: List = None) -> ArrayLike:
        x2d, squeeze = self._as_2d(x)
        cols = self._resolve_cols(cols)
        z = x2d.clone() if torch.is_tensor(x2d) else x2d.copy()
        for local_i, global_i in enumerate(cols):
            spec = self.specs[global_i]
            z[:, local_i] = spec.user_transform.physical_to_unit(x2d[:, local_i])
        return self._restore_shape(z, squeeze)

    def unit_to_physical_user(self, z: ArrayLike, cols: List = None) -> ArrayLike:
        z2d, squeeze = self._as_2d(z)
        cols = self._resolve_cols(cols)
        x = z2d.clone() if torch.is_tensor(z2d) else z2d.copy()
        for local_i, global_i in enumerate(cols):
            spec = self.specs[global_i]
            x[:, local_i] = spec.user_transform.unit_to_physical(z2d[:, local_i])
        return self._restore_shape(x, squeeze)

    def get_physical_bounds(self, as_tensor: bool = False) -> ArrayLike:
        bounds = np.stack([self._lower_bound, self._upper_bound], axis=0)
        return torch.tensor(bounds, dtype=torch.float64) if as_tensor else bounds



def build_transformer(config) -> ParameterTransformer:
   return ParameterTransformer(
        parameter_names=config.PARAMETER_NAMES,
        parameter_bounds=config.PARAMETER_BOUNDS,
        parameter_transforms=config.PARAMETER_TRANSFORMATION,
    )