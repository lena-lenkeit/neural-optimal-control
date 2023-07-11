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
import optimal_control.solvers as solvers
from optimal_control.utils import exists


def invert(x):
    return -x


class DirectSolverState(solvers.SolverState):
    optimizer_state: optax.OptState


class DirectSolver(solvers.AbstractSolver):
    optimizer: optax.GradientTransformation
    ignore_nans: bool = False

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
        constraint_chain: constraints.ConstraintChain,
        control: controls.AbstractControl,
        key: jax.random.KeyArray,
    ) -> Tuple[DirectSolverState, controls.AbstractControl, float]:
        # Get reward & gradients w.r.t. control
        grad_fn = eqx.filter_value_and_grad(solvers.evaluate_reward, has_aux=True)

        (reward, control), control_grads = grad_fn(
            control,
            constraint_chain=constraint_chain,
            environment=environment,
            environment_state=environment_state,
            reward_fn=reward_fn,
            key=key,
        )

        if self.ignore_nans:
            control_grads = jax.tree_map(
                lambda x: jnp.where(jnp.isnan(x), jnp.zeros_like(x), x),
                control_grads,
            )

        # Apply optimizer transformations
        control_params, control_static = eqx.partition(control, eqx.is_array)
        updates, optimizer_state = self.optimizer.update(
            control_grads, state.optimizer_state, params=control_params
        )

        # Flip gradient sign, to maximize the reward
        updates = jax.tree_map(invert, updates)

        # Update control parameters
        control_params = optax.apply_updates(control_params, updates)
        control = eqx.combine(control_params, control_static)

        return DirectSolverState(optimizer_state=optimizer_state), control, reward
