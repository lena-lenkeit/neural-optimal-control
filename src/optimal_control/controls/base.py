import abc
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, PRNGKeyArray, PyTree, Scalar

from optimal_control.utils import exists


class MemoryOutput(eqx.Module):
    derivative: Optional[PyTree] = None
    next: Optional[PyTree] = None


class ControlOutput(eqx.Module):
    values: PyTree
    memory: Optional[MemoryOutput] = None


class AbstractControl(eqx.Module):
    @abc.abstractmethod
    def __call__(self, **kwargs) -> ControlOutput:
        ...


class AbstractContinuousControl(AbstractControl):
    @abc.abstractmethod
    def get_control_values(
        self,
        *,
        t: Optional[Scalar] = None,
        y: Optional[PyTree] = None,
        args: Optional[PyTree] = None,
        dy_dt: Optional[PyTree] = None,
        **kwargs,
    ) -> ControlOutput:
        ...

    @abc.abstractmethod
    def modify_y0(self, t0: Scalar, y0: PyTree, args: PyTree) -> PyTree:
        ...


class AbstractDiscreteControl(AbstractControl):
    @abc.abstractmethod
    def step_control_values(
        self,
        *,
        t0: Optional[Scalar] = None,
        t1: Optional[Scalar] = None,
        y: Optional[PyTree] = None,
        args: Optional[PyTree] = None,
        dy_dt: Optional[PyTree] = None,
        memory: Optional[PyTree] = None,
        **kwargs,
    ) -> ControlOutput:
        ...

    @abc.abstractmethod
    def modify_y0(self, t0: Scalar, y0: PyTree, args: PyTree) -> PyTree:
        ...

    @abc.abstractmethod
    def initialize_memory(self, t0: Scalar, y0: PyTree, args: PyTree) -> PyTree:
        ...


class TimeDependentControl(AbstractControl):
    control_fn: Callable[[Scalar], PyTree]

    def __call__(self, *, t: Scalar, **kwargs) -> ControlOutput:
        return ControlOutput(self.control_fn(t))


class AbstractConstrainableControl(AbstractControl):
    def apply_constraint(
        self, constraint: Callable[[PyTree], PyTree]
    ) -> "AbstractConstrainableControl":
        ...


class StateDependentControl(AbstractControl):
    control_fn: Callable[[Scalar], PyTree]

    def __call__(self, *, y: PyTree, **kwargs) -> ControlOutput:
        return ControlOutput(self.control(y))


class AbstractCDERNNControl(AbstractContinuousControl):
    def get_control_values(
        self,
        *,
        t: Optional[Scalar] = None,
        system_y: Optional[PyTree] = None,
        control_y: Optional[PyTree] = None,
        args: Optional[PyTree] = None,
        dy_dt: Optional[PyTree] = None,
        **kwargs,
    ) -> ControlOutput:
        # Construct the path derivative
        dy_dt = self.get_path_derivative(system_y, dy_dt)
        dX_dt = jnp.concatenate((jnp.ones(1), dy_dt))

        # Get the latent derivatives
        dz_dt_mat = self.get_latent_derivative_matrix(z)
        dz_dt = dz_dt_mat @ dX_dt

        # Map latents to controls
        z = control_y
        c = self.map_latents_to_controls(z)

        return ControlOutput(values=c, memory=MemoryOutput(derivative=dz_dt))

    @abc.abstractmethod
    def get_latent_derivative_matrix(
        self, z: Float[Array, "latents"]
    ) -> Float[Array, "latents states"]:
        ...

    @abc.abstractmethod
    def map_latents_to_controls(self, z: Float[Array, "latents"]) -> PyTree:
        ...

    @abc.abstractmethod
    def get_path_derivative(self, y: PyTree, dy_dt: PyTree) -> Float[Array, "latents"]:
        ...

    @abc.abstractmethod
    def get_initial_latents(
        self, t0: Scalar, y0: PyTree, args: PyTree
    ) -> Float[Array, "latents"]:
        ...

    def extend_y0(self, t0: Scalar, y0: PyTree, args: PyTree) -> Array:
        return self.get_initial_latents(t0, y0, args)


class ODERNNControl(AbstractDiscreteControl):
    ...


class StepRNNControl(AbstractDiscreteControl):
    ...


# Some sort of lambda-ify auto-transformation?


# Helpers
def log_derivative(y: Array, dy_dt: Array) -> Array:
    return y / dy_dt


# def auto_control()
