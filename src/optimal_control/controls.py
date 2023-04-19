from functools import partial
from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from optimal_control.utils import exists


class AbstractControl(eqx.Module):
    r""""""


class LambdaControl(AbstractControl):
    control_fun: Callable[[ArrayLike], Array]

    def __call__(self, t: ArrayLike) -> Array:
        return self.control_fun(t)


class InterpolationControl(AbstractControl):
    control: Array
    channels: int
    steps: int
    t_start: float
    t_end: float
    method: str

    def __init__(
        self,
        channels: int,
        steps: int,
        t_start: float,
        t_end: float,
        method: str = "linear",
        control: Optional[Array] = None,
    ):
        self.channels = channels
        self.steps = steps
        self.t_start = t_start
        self.t_end = t_end
        self.method = method

        if not exists(control):
            self.control = jnp.zeros((self.steps, self.channels))
        else:
            self.control = control

    @staticmethod
    def interpolate_linear(x: ArrayLike, xp: ArrayLike, fp: ArrayLike) -> Array:
        vintp = jax.vmap(
            partial(jnp.interp, left=0.0, right=0.0),
            in_axes=(None, None, -1),
            out_axes=-1,
        )

        return vintp(x, xp, fp)

    @staticmethod
    def interpolate_step(x: ArrayLike, xp: ArrayLike, fp: ArrayLike) -> Array:
        def interp(x: ArrayLike, xp: ArrayLike, fp: ArrayLike) -> Array:
            idx = jnp.searchsorted(xp, x)
            y = jnp.where((x <= xp[0]) | (x > xp[-1]), 0.0, fp[idx - 1])

            return y

        vintp = jax.vmap(interp, in_axes=(None, None, -1), out_axes=-1)
        return vintp(x, xp, fp)

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
