import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import PyTree

EnvironmentState = PyTree


class AbstractEnvironment(eqx.Module):
    ...
