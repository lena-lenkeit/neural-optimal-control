import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, PyTree

import optimal_control.environments as environments
import optimal_control.solvers as solvers

EnvironmentState = PyTree


class AbstractEnvironment(eqx.Module):
    def init(self, *args, **kwargs) -> EnvironmentState:
        ...

    def integrate(
        self, control: solvers.AbstractControl, state: environments.EnvironmentState
    ) -> Array:
        ...
