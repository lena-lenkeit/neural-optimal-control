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
import optimal_control.controls as controls
import optimal_control.environments as environments
from optimal_control.utils import exists


class AbstractSolver(eqx.Module):
    @abc.abstractmethod
    def init(self, *args, **kwargs):
        r""""""

    @abc.abstractmethod
    def step(self, *args, **kwargs):
        r""""""


class DirectSolver(AbstractSolver):
    def init(
        self,
        optimizer: optax.GradientTransformation,
        control: controls.AbstractControl,
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
        _constraints: List[constraints.AbstractConstraint],
        control: controls.AbstractControl,
    ) -> Tuple[ArrayLike, controls.AbstractControl, optax.OptState]:
        @jax.value_and_grad
        def _reward(params, static, rewards, environment, environment_state):
            control = eqx.combine(params, static)
            env_seq = environment.integrate(control, environment_state)
            reward = -rewards(env_seq)

            return reward

        def _ensure_constraints(
            control: controls.AbstractControl,
            _constraints: List[constraints.AbstractConstraint],
        ) -> controls.AbstractControl:
            # Ensure validity of constraints
            for constraint in _constraints:
                control = eqx.tree_at(
                    lambda control: control.control,
                    control,
                    replace_fn=constraint.project,
                )

            return control

        control = _ensure_constraints(control, _constraints)

        params, static = eqx.partition(control, eqx.is_array)
        reward, grads = _reward(params, static, rewards, environment, environment_state)

        updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
        params = optax.apply_updates(params, updates)

        control = eqx.combine(params, static)
        control = _ensure_constraints(control, _constraints)

        return -reward, control, optimizer_state
