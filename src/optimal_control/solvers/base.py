import abc
from typing import Callable, List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree, Scalar

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


"""
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
"""


def evaluate_reward(
    control: controls.AbstractControl,
    constraint_chain: List[constraints.AbstractConstraint],
    environment: environments.AbstractEnvironment,
    environment_state: environments.EnvironmentState,
    reward_fn: Callable[[PyTree], float],
    num_control_points: int,
    key: jax.random.KeyArray,
) -> Tuple[Scalar, controls.AbstractControl]:
    # 1. Build the control for the environment

    eval_control = None
    ret_control = None
    try_project = True

    # Check if this control implicitly encodes another control
    get_implicit_control_fn = getattr(control, "get_implicit_control", None)
    if exists(get_implicit_control_fn):
        control = get_implicit_control_fn()

        # Stop the implicitly encoded control from being projected
        try_project = False

    # 2. Try to bake constraints into the control
    # Project -> Transform -> Penalty

    # Try to project the parameters of the control, such that the outputs lie in the
    # constrained region
    could_project = False
    if try_project:
        apply_projection_fn = getattr(control, "apply_projection", None)

        if exists(apply_projection_fn):
            for constraint in constraint_chain:
                apply_projection_fn = getattr(control, "apply_projection")
                control = apply_projection_fn(constraint.project)

            could_project = True

    # Try to transform the control outputs, such that they lie in the constrained region
    could_transform = False
    if not could_project:
        # In this case, we need to cache the control onto interpolation curves
        ...

    # Try to create penalty terms for each constraint
    if not could_transform:
        ...
