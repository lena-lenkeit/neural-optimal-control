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
        key: jax.random.KeyArray,
    ) -> Tuple[ArrayLike, controls.AbstractControl, optax.OptState]:
        def _apply_constraint_transforms(
            control: controls.AbstractControl,
            _constraints: List[constraints.AbstractConstraint],
        ) -> controls.AbstractControl:
            # Apply transforms
            for constraint in _constraints:
                control = eqx.tree_at(
                    lambda control: control.control,
                    control,
                    replace_fn=constraint.transform,
                )

            return control

        @jax.value_and_grad
        def _reward(
            params, static, rewards, environment, environment_state, _constraints, key
        ):
            control: controls.AbstractControl = eqx.combine(params, static)
            # control = _apply_constraint_transforms(control, _constraints)

            # Evaluate control
            num_points = 10
            points = jnp.linspace(
                control.t_start, control.t_end, num=num_points, endpoint=False
            )
            spacing = (control.t_end - control.t_start) / num_points
            points += spacing / 2

            # Transform control
            full_control = jax.vmap(control)(points.reshape(num_points, 1))
            full_control = _constraints[0].transform(full_control)

            # Package control
            control = controls.InterpolationControl(
                full_control.shape[1],
                full_control.shape[0],
                control.t_start,
                control.t_end,
                method="step",
                control=full_control,
            )

            env_seq = environment.integrate(control, environment_state, key)
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

        # control = _ensure_constraints(control, _constraints)

        params, static = eqx.partition(control, eqx.is_array)
        # key, subkey = jax.random.split(key)
        reward, grads = _reward(
            params, static, rewards, environment, environment_state, _constraints, key
        )

        updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
        params = optax.apply_updates(params, updates)

        control = eqx.combine(params, static)
        # control = _ensure_constraints(control, _constraints)

        return -reward, control, optimizer_state
