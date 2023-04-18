import abc
from functools import partial
from typing import Any, Callable, List

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import optimal_control.constraints as constraints
import optimal_control.environments as environments


class AbstractSolver(eqx.Module):
    @abc.abstractmethod
    def init(self, *args, **kwargs):
        r""""""

    @abc.abstractmethod
    def step(self, *args, **kwargs):
        r""""""


class AbstractControl(eqx.Module):
    r""""""


class InterpolationControl(AbstractControl):
    control: Array
    channels: int
    steps: int
    t_start: float
    t_end: float
    method: str = "linear"

    def __init__(self, channels: int, steps: int, t_start: float, t_end: float, key):
        self.control = jnp.zeros((steps, channels))

    @staticmethod
    def interpolate_linear(x: ArrayLike, xp: ArrayLike, fp: ArrayLike) -> Array:
        vintp = jax.vmap(jnp.interp, in_axes=(None, None, -1), out_axes=-1)
        return vintp(x, xp, fp, left=0.0, right=0.0)

    @staticmethod
    def interpolate_step(x: ArrayLike, xp: ArrayLike, fp: ArrayLike) -> Array:
        def interp(x: ArrayLike, xp: ArrayLike, fp: ArrayLike) -> Array:
            idx = jnp.searchsorted(xp, x)
            y = jnp.where(idx == 0, 0.0, fp[idx - 1])

            return y

        vintp = jax.vmap(interp, in_axes=(None, None, -1), out_axes=-1)
        return vintp(x, xp, fp, left=0.0, right=0.0)

    @staticmethod
    def interpolate(x: ArrayLike, xp: ArrayLike, fp: ArrayLike, method: str) -> Array:
        if method == "linear":
            return InterpolationControl.interpolate_linear(x, xp, fp)
        elif method == "step":
            return InterpolationControl.interpolate_step(x, xp, fp)

    def __call__(self, t: ArrayLike) -> Array:
        t = (t - self.t_start) / (self.t_end - self.t_start)
        return InterpolationControl.interpolate(
            t, jnp.linspace(0.0, 1.0, self.steps), self.control, self.method
        )


class ImplicitControl(AbstractControl):
    mlp: eqx.Module


class DirectSolver(AbstractSolver):
    def step(
        environment: environments.AbstractEnvironment,
        rewards: Callable[[Array], ArrayLike],
        constraints: List[constraints.AbstractConstraint],
        control: AbstractControl,
    ) -> AbstractControl:
        pass
