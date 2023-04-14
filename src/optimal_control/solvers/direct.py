import abc
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float


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

    def __init__(self, channels: int, steps: int, t_start: float, t_end: float, key):
        self.control = jnp.zeros((steps, channels))

    @staticmethod
    def interpolate_linear(x: ArrayLike, xp: ArrayLike, fp: ArrayLike) -> Array:
        vintp = jax.vmap(jnp.interp, in_axes=(None, None, -1), out_axes=-1)
        return vintp(x, xp, fp)

    # @staticmethod
    # def interpolate_step()

    def __call__(self, t: ArrayLike) -> Array:
        t = (t - self.t_start) / (self.t_end - self.t_start)
        return InterpolationControl.interpolate(
            t, jnp.linspace(0.0, 1.0, self.steps), self.control
        )


class ImplicitControl(AbstractControl):
    mlp: eqx.Module


class DirectSolver(AbstractSolver):
    control: AbstractControl

    def init(
        self,
        num_controls: int,
        control_start: float,
        control_end: float,
        control_steps: int,
    ):
        return
