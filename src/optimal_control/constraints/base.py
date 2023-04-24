import abc

import diffrax
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import optimal_control.controls as controls


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

    def transform(self, control: Array) -> Array:
        raise NotImplementedError()

    def transform_continuous(
        self, control: controls.AbstractControl
    ) -> controls.LambdaControl:
        ## Via continuous softmax

        # Calculate the softmax normalization factor
        terms = diffrax.ODETerm(lambda t, x, args: jnp.exp(args(t)))
        solver = diffrax.Dopri5()
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=0.0)

        softmax_denominator = diffrax.diffeqsolve(
            terms=terms,
            solver=solver,
            t0=0.0,
            t1=1.0,
            dt0=None,
            y0=jnp.zeros_like(control(jnp.asarray([0.0]))),
            args=control,
            saveat=diffrax.SaveAt(t1=True),
            max_steps=10000,
            adjoint=diffrax.RecursiveCheckpointAdjoint(checkpoints=10000),
        ).ys[-1]

        # Construct the transformed control
        transformed_control = controls.LambdaControl(
            lambda t: jnp.exp(control(t)) / softmax_denominator
        )

        return transformed_control

    def penalty(self, control: Array) -> ArrayLike:
        raise NotImplementedError()

    def is_instantaneous(self) -> bool:
        return False
