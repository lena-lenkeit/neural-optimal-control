import abc
from functools import partial
from typing import Callable, List, Literal, Optional, Tuple, Union

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
        constrained_nodes = constraint_fn(
            self.curve.nodes, jax.lax.stop_gradient(self.curve.times)
        )
        constrained_curve = nn.InterpolationCurve(
            method=self.curve.method,
            nodes=constrained_nodes,
            times=self.curve.times,
            has_even_spacing=self.curve.has_even_spacing,
        )

        return InterpolationCurveControl(constrained_curve)


class ImplicitTemporalControl(controls.AbstractControl):
    implicit_fn: eqx.Module
    t_start: Scalar
    t_end: Scalar
    to_curve: bool
    curve_times: Optional[Array] = None
    curve_interpolation: Optional[Literal["step", "linear"]] = None
    curve_steps: Optional[int] = None

    def normalize_time(self, t: Array) -> Array:
        return (t - self.t_start) / (self.t_end - self.t_start)

    def __call__(self, t: Scalar, **kwargs) -> controls.ControlOutput:
        return self.implicit_fn(self.normalize_time(t))

    def get_implicit_control(self) -> InterpolationCurveControl:
        if not self.to_curve:
            return None

        if exists(self.curve_times):
            curve_times = self.normalize_time(self.curve_times).reshape(-1, 1)
            curve_values = jax.vmap(self.implicit_fn)(curve_times)

            curve = nn.InterpolationCurve(
                method=self.curve_interpolation, nodes=curve_values, times=curve_times
            )

        elif exists(self.curve_steps):
            times = jnp.linspace(-1.0, 1.0, num=self.curve_steps).reshape(-1, 1)
            curve_values = jax.vmap(self.implicit_fn)(times)
            curve_channels = eqx.filter_eval_shape(self.implicit_fn, times[0]).shape[-1]

            curve = nn.InterpolationCurve(
                method=self.curve_interpolation,
                nodes=curve_values,
                t_start=self.t_start,
                t_end=self.t_end,
                steps=self.curve_steps,
                channels=curve_channels,
            )

        else:
            raise TypeError(
                "One of curve_times or curve_steps must be specified to construct an implicit control"
            )

        return InterpolationCurveControl(curve)
