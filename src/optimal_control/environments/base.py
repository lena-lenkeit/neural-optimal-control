import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, PyTree

import optimal_control.controls as controls

# EnvironmentState = PyTree


class EnvironmentState(eqx.Module):
    ...


class AbstractEnvironment(eqx.Module):
    def init(self, *args, **kwargs) -> EnvironmentState:
        ...

    def integrate(
        self,
        control: controls.AbstractControl,
        state: EnvironmentState,
        key: jax.random.KeyArray,
    ) -> PyTree:
        ...
