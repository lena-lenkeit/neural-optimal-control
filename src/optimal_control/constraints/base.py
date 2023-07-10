import abc
from functools import partial
from typing import Optional, Sequence

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree, Scalar

import optimal_control.controls as controls


class AbstractConstraint(eqx.Module):
    ...


class AbstractProjectionConstraint(AbstractConstraint):
    @abc.abstractmethod
    def project(self, control: PyTree) -> PyTree:
        ...


class AbstractLocalTransformationConstraint(AbstractConstraint):
    @abc.abstractmethod
    def transform_single(self, control: PyTree) -> PyTree:
        ...

    def transform_series(self, control: PyTree) -> PyTree:
        return jax.vmap(self.transform_single)(control)


class AbstractGlobalTransformationConstraint(AbstractConstraint):
    @abc.abstractmethod
    def transform_series(self, control: PyTree) -> PyTree:
        ...


class AbstractPenaltyConstraint(AbstractConstraint):
    penalty_weight: Optional[Scalar] = None


class AbstractLocalPenaltyConstraint(AbstractPenaltyConstraint):
    @abc.abstractmethod
    def penalty_single(self, values: PyTree) -> PyTree:
        ...

    def penalty_series(self, values: PyTree) -> PyTree:
        return jax.vmap(self.penalty_single)(values)

    def penalty_ode_term(self, values: PyTree) -> PyTree:
        return self.penalty_single(values)

    def penalty_ode_reward(self, integrated_term: PyTree) -> Scalar:
        return integrated_term * self.penalty_weight


class AbstractGlobalPenaltyConstraint(AbstractPenaltyConstraint):
    @abc.abstractmethod
    def penalty_series(self, values: PyTree) -> PyTree:
        ...

    @abc.abstractmethod
    def penalty_ode_term(self, values: PyTree) -> PyTree:
        ...

    @abc.abstractmethod
    def penalty_ode_reward(self, integrated_term: PyTree) -> Scalar:
        ...


class ConstraintChain(eqx.Module):
    projections: List[AbstractProjectionConstraint]
    transformations: List[AbstractGlobalTransformationConstraint]
    penalties: List[AbstractPenaltyConstraint]


class NonNegativeConstantIntegralConstraint(AbstractConstraint):
    integral: ArrayLike
    eps: ArrayLike = 1e-10

    def project(self, control: Array) -> Array:
        # Non-zero constraint via clipping, with eps to prevent divide-by-zero
        control = jnp.where(control < self.eps, self.eps, control)

        # Normalize to constant integral by rescaling
        control_integral = jnp.mean(control, axis=0)
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


class ConstantIntegralConstraint(AbstractConstraint):
    """
    Restricts the integral of the control signal by rescaling to match a target value
    """

    integral: ArrayLike

    def project(self, control: Array) -> Array:
        raise NotImplementedError()

    def transform(self, control: Array) -> Array:
        # Evaluate integral
        integral = jnp.mean(control, axis=0, keepdims=True)

        # Rescale to match target integral
        factor = self.integral / integral
        return control * factor

    def penalty(self, control: Array) -> ArrayLike:
        raise NotImplementedError()

    def is_instantaneous(self) -> bool:
        return False


class ConvolutionConstraint(AbstractConstraint):
    kernel: Array
    padding_type: str
    pad_left: int
    pad_right: int

    @staticmethod
    def clip_convolve(
        a: ArrayLike, v: ArrayLike, pad_left: int, pad_right: int
    ) -> ArrayLike:
        a = jnp.concatenate(
            (jnp.full(pad_left, a[0]), a, jnp.full(pad_right, a[-1])), axis=0
        )
        c = jnp.convolve(a, v, mode="same")

        return c[pad_left : len(c) - pad_right]

    def project(self, control: Array) -> Array:
        raise NotImplementedError()

    def transform(self, control: Array) -> Array:
        # Convolve with kernel

        if self.padding_type == "zero":
            return jax.vmap(
                partial(jnp.convolve, mode="same"), in_axes=(1, 1), out_axes=1
            )(control, self.kernel)
        elif self.padding_type == "clip":
            return jax.vmap(
                partial(
                    ConvolutionConstraint.clip_convolve,
                    pad_left=self.pad_left,
                    pad_right=self.pad_right,
                ),
                in_axes=(1, 1),
                out_axes=1,
            )(control, self.kernel)

    def penalty(self, control: Array) -> ArrayLike:
        raise NotImplementedError()

    def is_instantaneous(self) -> bool:
        return False
