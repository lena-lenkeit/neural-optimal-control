import abc
from functools import partial
from typing import Any, Callable, List, Optional, Tuple

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, ArrayLike, PyTree

import optimal_control.constraints as constraints
import optimal_control.controls as controls
import optimal_control.environments as environments
from optimal_control.utils import exists


class SolverState(eqx.Module):
    ...


class AbstractSolver(eqx.Module):
    @abc.abstractmethod
    def init(self, control: controls.AbstractControl) -> SolverState:
        ...

    @abc.abstractmethod
    def step(
        self,
        state: SolverState,
        environment: environments.AbstractEnvironment,
        environment_state: environments.EnvironmentState,
        reward_fn: Callable[[PyTree], float],
        constraint_chain: List[constraints.AbstractConstraint],
        control: controls.AbstractControl,
        key: jax.random.KeyArray,
    ) -> Tuple[SolverState, controls.AbstractControl, float]:
        ...


def apply_constraint_chain(
    control: Array, constraint_chain: List[constraints.AbstractConstraint]
) -> Array:
    for constraint in constraint_chain:
        control = constraint.transform(control)

    return control


def build_control(
    control: controls.AbstractControl,
    constraint_chain: List[constraints.AbstractConstraint],
    num_points: int,
) -> controls.InterpolationControl:
    # Evaluate control
    points = jnp.linspace(
        control.t_start, control.t_end, num=num_points, endpoint=False
    )
    spacing = (control.t_end - control.t_start) / num_points
    points += spacing / 2

    # Transform control
    control_values = jax.vmap(control)(points.reshape(num_points, 1))
    control_values = apply_constraint_chain(control_values, constraint_chain)

    # Package control
    interpolation_control = controls.InterpolationControl(
        channels=control_values.shape[1],
        steps=control_values.shape[0],
        t_start=control.t_start,
        t_end=control.t_end,
        method="step",
        control=control_values,
    )

    return interpolation_control


def evaluate_reward(
    control: controls.AbstractControl,
    constraint_chain: List[constraints.AbstractConstraint],
    environment: environments.AbstractEnvironment,
    environment_state: environments.EnvironmentState,
    reward_fn: Callable[[PyTree], float],
    num_control_points: int,
    key: jax.random.KeyArray,
) -> float:
    control = build_control(
        control=control,
        constraint_chain=constraint_chain,
        num_points=num_control_points,
    )

    environment_output = environment.integrate(
        control=control, state=environment_state, key=key
    )

    reward = reward_fn(environment_output)
    return reward


class DirectSolverState(SolverState):
    optimizer_state: optax.OptState


class DirectSolver(AbstractSolver):
    optimizer: optax.GradientTransformation
    num_control_points: int

    def init(self, control: controls.AbstractControl) -> DirectSolverState:
        control_params = eqx.filter(control, eqx.is_array)
        optimizer_state = self.optimizer.init(control_params)

        return DirectSolverState(optimizer_state=optimizer_state)

    def step(
        self,
        state: DirectSolverState,
        environment: environments.AbstractEnvironment,
        environment_state: environments.EnvironmentState,
        reward_fn: Callable[[PyTree], float],
        constraint_chain: List[constraints.AbstractConstraint],
        control: controls.AbstractControl,
        key: jax.random.KeyArray,
    ) -> Tuple[DirectSolverState, controls.AbstractControl, float]:
        grad_fn = eqx.filter_value_and_grad(evaluate_reward)

        reward, control_grads = grad_fn(
            control,
            constraint_chain=constraint_chain,
            environment=environment,
            environment_state=environment_state,
            reward_fn=reward_fn,
            num_control_points=self.num_control_points,
            key=key,
        )

        control_params, control_static = eqx.partition(control, eqx.is_array)
        updates, optimizer_state = self.optimizer.update(
            control_grads, state.optimizer_state, params=control_params
        )

        control_params = optax.apply_updates(control_params, updates)
        control = eqx.combine(control_params, control_static)

        return DirectSolverState(optimizer_state=optimizer_state), control, reward
