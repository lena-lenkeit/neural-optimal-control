import abc
import logging
from typing import Callable, List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PRNGKeyArray, PyTree, Scalar

import optimal_control.constraints as constraints
import optimal_control.controls as controls
import optimal_control.environments as environments
from optimal_control.utils import exists


class SolverState(eqx.Module):
    ...


class AbstractSolver(eqx.Module):
    @abc.abstractmethod
    def init(self, control: controls.AbstractControl, key: PRNGKeyArray) -> SolverState:
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
        integrate_kwargs: Optional[dict] = None,
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


def build_control(
    base_control: controls.AbstractControl,
    chain: constraints.ConstraintChain,
) -> Tuple[controls.AbstractControl, controls.AbstractControl]:
    control = base_control

    def project(
        control: controls.AbstractConstrainableControl,
        projections: List[constraints.AbstractProjectionConstraint],
    ) -> controls.AbstractConstrainableControl:
        for constraint in projections:
            control = control.apply_constraint(constraint.project)

        return control

    def transform(
        control: controls.AbstractConstrainableControl,
        transformations: List[constraints.AbstractGlobalTransformationConstraint],
    ) -> controls.AbstractConstrainableControl:
        for constraint in transformations:
            control = control.apply_constraint(constraint.transform_series)

        return control

    implicit_control_fn = getattr(control, "get_implicit_control", None)
    if exists(implicit_control_fn):
        implicit_control = implicit_control_fn()

        if exists(implicit_control):
            carry_control = control
            control = implicit_control
        else:
            control = project(control, chain.projections)
            carry_control = control
    else:
        control = project(control, chain.projections)
        carry_control = control

    control = transform(control, chain.transformations)

    """
    if len(penalty_constraints) > 0:

        def penalty_ode(
            t: Scalar,
            f_y: PyTree,
            g_y: PyTree,
            u: PyTree,
            args: PyTree,
            penalty_constraints: List[constraints.AbstractConstraint],
        ) -> PyTree:
            for constraint in penalty_constraints:
                constraint.penalty()

    else:
        penalty_ode = None
    """

    return control, carry_control  # , penalty_ode


def evaluate_reward(
    control: controls.AbstractControl,
    constraint_chain: constraints.ConstraintChain,
    environment: environments.AbstractEnvironment,
    environment_state: environments.EnvironmentState,
    reward_fn: Callable[[PyTree], float],
    key: jax.random.KeyArray,
    integrate_kwargs: Optional[dict] = None,
) -> Tuple[Scalar, controls.AbstractControl]:
    # 1. Build the control for the environment
    control, projected_control = build_control(control, constraint_chain)

    # 2. Integrate the environment
    environment_output = environment.integrate(
        control=control, state=environment_state, key=key, **integrate_kwargs
    )

    # 3. Calculate reward
    reward = reward_fn(environment_output)
    return reward, projected_control
