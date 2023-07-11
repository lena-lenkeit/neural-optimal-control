import abc
from functools import partial
from typing import List, Literal, Optional, Sequence

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

    def project(self, values: PyTree, times: PyTree) -> PyTree:
        def map_fn(values: Array, times: Array) -> Array:
            # Non-negativity constraint via clipping
            values = jax.tree_util.tree_map(
                lambda x: jnp.clip(x, a_min=self.eps), values
            )

            # Calculate integral by summing over constant pieces
            area = values * inner_dt_from_times(times).reshape(-1, 1)
            integral = jnp.sum(area, axis=0, keepdims=True)

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
            # Normalize via area-scaled, numerically-stable softmax
            dt = inner_dt_from_times(times).reshape(-1, 1)

            max_value = values.max(axis=0, keepdims=True)
            stable_exp = jnp.exp(values - jax.lax.stop_gradient(max_value))
            area = stable_exp * dt
            normalized = stable_exp / (jnp.sum(area, axis=0, keepdims=True))

            if self.norm == "average":
                normalized = normalized * outer_dt_from_times(times).reshape(-1, 1)

            # Rescale to target
            values = normalized * self.target

            return values

        values = jax.tree_util.tree_map(map_fn, values, times)
        return values


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
