import abc
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, PRNGKeyArray, PyTree, Scalar

import optimal_control.controls as controls
import optimal_control.nn as nn
from optimal_control.utils import exists


class LambdaControl(controls.AbstractControl):
    control_fn: Union[Callable[[PyTree, PyTree], Array], Callable[[PyTree], Array]]
    data: Optional[PyTree] = None

    def __call__(self, **kwargs) -> Array:
        if exists(self.data):
            return self.control_fn(kwargs, self.data)
        else:
            return self.control_fn(kwargs)


class InterpolationCurveControl(controls.AbstractConstrainableControl):
    curve: nn.InterpolationCurve

    def __call__(self, t: Scalar, **kwargs) -> controls.ControlOutput:
        return self.curve(t)

    def apply_constraint(
        self, constraint_fn: Callable[[PyTree, Optional[PyTree]], PyTree]
    ) -> "InterpolationCurveControl":
        constrained_nodes = constraint_fn(self.curve.nodes, self.curve.times)
        constrained_curve = nn.InterpolationCurve(
            method=self.curve.method,
            nodes=constrained_nodes,
            times=self.curve.times,
            has_even_spacing=self.curve.has_even_spacing,
        )

        return InterpolationCurveControl(constrained_curve)