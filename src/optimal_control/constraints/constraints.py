import abc
from functools import partial
from typing import List, Literal, Optional, Sequence, Tuple, Union

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree, Scalar

from optimal_control.constraints.base import (
    AbstractConstraint,
    AbstractGlobalPenaltyConstraint,
    AbstractGlobalTransformationConstraint,
    AbstractProjectionConstraint,
)


def inner_dt_from_times(times: Array) -> Array:
    return times[1:] - times[:-1]


def outer_dt_from_times(times: Array) -> Scalar:
    return times[-1] - times[0]


class NonNegativeConstantIntegralConstraint(
    AbstractProjectionConstraint, AbstractGlobalTransformationConstraint
):
    target: PyTree
    norm: Literal["average", "integral"] = "average"
    eps: Scalar = jnp.full(1, 1e-10)
    constrain_sum: bool = False

    def project(self, values: PyTree, times: PyTree) -> PyTree:
        def map_fn(values: Array, times: Array) -> Array:
            # Non-negativity constraint via clipping
            values = jax.tree_util.tree_map(
                lambda x: jnp.clip(x, a_min=self.eps), values
            )

            if values.shape[0] == times.shape[0]:
                # Use the midpoint for linearly interpolated values
                interp_values = (values[1:] + values[:-1]) / 2

            else:
                # Use the left edge for constant values
                interp_values = values

            area = interp_values * inner_dt_from_times(times).reshape(-1, 1)

            # Calculate integral by summing over constant pieces
            integral = jnp.sum(area, axis=0, keepdims=True)

            if self.constrain_sum:
                integral = jnp.sum(integral)

            # Average over the time interval, if necessary
            if self.norm == "average":
                integral = integral / outer_dt_from_times(times).reshape(-1, 1)

            # Normalize to target integral by rescaling
            values = values / integral * self.target

            return values

        values = jax.tree_util.tree_map(map_fn, values, times)
        return values

    def transform_series(self, values: PyTree, times: PyTree) -> PyTree:
        def map_fn(values: Array, times: Array) -> Array:
            """
            # Normalize via area-scaled, numerically-stable softmax
            dt = inner_dt_from_times(times).reshape(-1, 1)

            is_linear = values.shape[0] == times.shape[0]
            if is_linear:
                # Use the midpoint for linearly interpolated values
                interp_values = (values[1:] + values[:-1]) / 2

            else:
                # Use the left edge for constant values
                interp_values = values

            max_value = interp_values.max(axis=0, keepdims=True)
            stable_exp = jnp.exp(interp_values - jax.lax.stop_gradient(max_value))
            print(interp_values.shape, dt.shape, times.shape)
            area = stable_exp * dt
            # normalized = stable_exp / (jnp.sum(area, axis=0, keepdims=True))
            normalized = area / (jnp.sum(area, axis=0, keepdims=True))

            if self.norm == "average":
                normalized = normalized * outer_dt_from_times(times).reshape(-1, 1)

            # Rescale to target
            values = normalized * self.target

            # Add missing points for linear interpolation
            if is_linear:
                values = jnp.concatenate(
                    (values[:1], (values[1:] + values[:-1]) / 2, values[-1:]), axis=0
                )

            return values
            """

            def integrate(
                values: Array, times: Array, interpolation: Literal["step", "linear"]
            ) -> Array:
                dt = inner_dt_from_times(times)

                if interpolation == "step":
                    return jnp.sum(values * dt)
                elif interpolation == "linear":
                    midpoints = (values[1:] + values[:-1]) / 2
                    return jnp.sum(midpoints * dt)

            is_linear = values.shape[0] == times.shape[0]

            softmax = jax.nn.softmax(values, axis=(0, 1) if self.constrain_sum else 0)
            integral = jax.vmap(
                partial(integrate, interpolation="linear" if is_linear else "step"),
                in_axes=(-1, None),
                out_axes=-1,
            )(softmax, times)

            if self.constrain_sum:
                integral = jnp.sum(integral)

            normalized = softmax / integral * self.target

            if self.norm == "average":
                return normalized * outer_dt_from_times(times)
            elif self.norm == "integral":
                return normalized

        values = jax.tree_util.tree_map(map_fn, values, times)
        return values


class LimitedRangeConstantIntegralConstraint(AbstractGlobalTransformationConstraint):
    target: Array
    maximum: Array

    def transform_series(self, values: PyTree, times: PyTree) -> PyTree:
        def transform_fn(values: Array, times: Array) -> Array:
            # [-inf, +inf], undefined sum -> [0, 1], sum of one
            softmax = jax.nn.softmax(values, axis=0)

            # [0, 1], sum of one -> [0, +inf], correct integral
            # integral = jnp.mean(softmax, axis=0, keepdims=True)
            # values = softmax / integral
            values = softmax * self.target

            # [0, +inf] -> [0, maximum], correct integral
            for i in range(64):
                extra = jnp.clip(values - self.maximum, a_min=0)
                total = jnp.mean(extra, axis=0, keepdims=True)
                values = values - extra + total

            return values

        return jax.tree_map(transform_fn, values, times)


class ConstantIntegralConstraint(AbstractConstraint):
    """
    Restricts the integral of the control signal by rescaling to match a target value
    """

    integral: ArrayLike

    def project(self, control: Array) -> Array:
        raise NotImplementedError()

    def transform(self, control: Array) -> Array:
        # Evaluate integral
        integral = jnp.mean(control, axis=0, keepdims=True)

        # Rescale to match target integral
        factor = self.integral / integral
        return control * factor

    def penalty(self, control: Array) -> ArrayLike:
        raise NotImplementedError()

    def is_instantaneous(self) -> bool:
        return False


class ConvolutionConstraint(AbstractConstraint):
    kernel: Array
    padding_type: str
    pad_left: int
    pad_right: int

    @staticmethod
    def clip_convolve(
        a: ArrayLike, v: ArrayLike, pad_left: int, pad_right: int
    ) -> ArrayLike:
        a = jnp.concatenate(
            (jnp.full(pad_left, a[0]), a, jnp.full(pad_right, a[-1])), axis=0
        )
        c = jnp.convolve(a, v, mode="same")

        return c[pad_left : len(c) - pad_right]

    def project(self, control: Array) -> Array:
        raise NotImplementedError()

    def transform(self, control: Array) -> Array:
        # Convolve with kernel

        if self.padding_type == "zero":
            return jax.vmap(
                partial(jnp.convolve, mode="same"), in_axes=(1, 1), out_axes=1
            )(control, self.kernel)
        elif self.padding_type == "clip":
            return jax.vmap(
                partial(
                    ConvolutionConstraint.clip_convolve,
                    pad_left=self.pad_left,
                    pad_right=self.pad_right,
                ),
                in_axes=(1, 1),
                out_axes=1,
            )(control, self.kernel)

    def penalty(self, control: Array) -> ArrayLike:
        raise NotImplementedError()

    def is_instantaneous(self) -> bool:
        return False
