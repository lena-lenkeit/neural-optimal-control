import abc

import diffrax
import equinox as eqx
import jax
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
        ## Via discrete softmax

        return self.integral * jax.nn.softmax(control, axis=0) * control.shape[0]

    def transform_continuous(
        self, control: controls.AbstractControl
    ) -> controls.LambdaControl:
        ## Via continuous softmax

        # Calculate the softmax normalization factor (diffrax-based integration)
        """
        terms = diffrax.ODETerm(lambda t, x, args: jnp.exp(args(t.reshape(1))))
        solver = diffrax.Dopri5()
        # stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=0.0)

        softmax_denominator = diffrax.diffeqsolve(
            terms=terms,
            solver=solver,
            t0=0.0,
            t1=1.0,
            dt0=1.0 / 1000,
            y0=jnp.zeros_like(control(jnp.asarray([0.0]))),
            args=control,
            saveat=diffrax.SaveAt(t1=True),
            max_steps=1001,
            # stepsize_controller=stepsize_controller,
            adjoint=diffrax.RecursiveCheckpointAdjoint(checkpoints=1001),
        ).ys[-1]
        """

        # Calculate the softmax normalization factor (sum-based integration)
        softmax_denominator = jnp.mean(
            jnp.exp(jax.vmap(control)(jnp.linspace(0.0, 1.0, 1024).reshape(1024, 1))),
            axis=0,
        )

        # Construct the transformed control
        transformed_control = controls.LambdaControl(
            lambda t: self.integral * jnp.exp(control(t)) / softmax_denominator
        )

        return transformed_control

    def penalty(self, control: Array) -> ArrayLike:
        raise NotImplementedError()

    def is_instantaneous(self) -> bool:
        return False
