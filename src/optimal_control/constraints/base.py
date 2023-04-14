import abc

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


class AbstractConstraint(eqx.Module):
    @abc.abstractmethod
    def project(self, control: Array) -> Array:
        ...

    @abc.abstractmethod
    def transform(self, control: Array) -> Array:
        ...

    @abc.abstractmethod
    def penalty(self, control: Array) -> ArrayLike:
        ...

    @abc.abstractmethod
    def is_instantaneous(self) -> bool:
        ...


class NonNegativeConstantIntegralConstraint(AbstractConstraint):
    integral: ArrayLike
    eps: ArrayLike = 1e-10

    def project(self, control: Array) -> Array:
        # Non-zero constraint via clipping, with eps to prevent divide-by-zero
        control = jnp.where(control < self.eps, self.eps, control)

        # Normalize to constant integral by rescaling
        control_integral = jnp.sum(control, axis=0)
        control = (control / control_integral) * self.integral

        return control

    def is_instantaneous(self) -> bool:
        return False
