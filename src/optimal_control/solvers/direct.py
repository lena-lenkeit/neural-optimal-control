import abc
from functools import partial
from typing import Any, Callable, List, Optional, Tuple

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, ArrayLike

import optimal_control.constraints as constraints
import optimal_control.environments as environments
from optimal_control.utils import exists


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


class DirectSolver(AbstractSolver):
    def init(
        self,
        optimizer: optax.GradientTransformation,
        control: AbstractControl,
    ) -> optax.OptState:
        params, static = eqx.partition(control, eqx.is_array)
        optimizer_state = optimizer.init(params)

        return optimizer_state

    def step(
        self,
        optimizer_state: optax.OptState,
        optimizer: optax.GradientTransformation,
        environment_state: environments.EnvironmentState,
        environment: environments.AbstractEnvironment,
        rewards: Callable[[Array], ArrayLike],
        constraints: List[constraints.AbstractConstraint],
        control: AbstractControl,
    ) -> Tuple[ArrayLike, AbstractControl]:
        @jax.value_and_grad
        def _reward(params, static, environment_state):
            control = eqx.combine(params, static)
            env_seq = environment.integrate(control, environment_state)
            reward = rewards(env_seq)

            return reward

        def _ensure_constraints(
            control: AbstractControl, constraints: List[constraints.AbstractConstraint]
        ) -> AbstractControl:
            # Ensure validity of constraints
            for constraint in constraints:
                control = eqx.tree_at(
                    lambda control: control.control,
                    control,
                    replace_fn=constraint.project(),
                )

            return control

        control = _ensure_constraints(control, constraints)
        params, static = eqx.partition(control, eqx.is_array)
        reward, grads = _reward(params, static, environment_state)
        optimizer.update(grads, optimizer_state, params)
        control = _ensure_constraints(control, constraints)

        return reward, control
