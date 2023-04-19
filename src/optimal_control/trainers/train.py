from typing import Any, Callable, List, Tuple

import equinox as eqx
import optax
from jaxtyping import Array, ArrayLike
from tqdm.auto import tqdm as tq
from tqdm.auto import trange

import optimal_control.constraints as constraints
import optimal_control.controls as controls
import optimal_control.environments as environments
import optimal_control.solvers as solvers


def solve_optimal_control_problem(
    environment: environments.AbstractEnvironment,
    rewards: Callable[[Array], ArrayLike],
    _constraints: List[constraints.AbstractConstraint],
    solver: solvers.AbstractSolver,
    control: controls.AbstractControl,
    num_steps: int,
):
    def _init(
        environment: environments.AbstractEnvironment,
        solver: solvers.AbstractSolver,
        control: controls.AbstractControl,
        optimizer: optax.GradientTransformation,
    ) -> Tuple[environments.EnvironmentState, optax.OptState]:
        environment_state = environment.init()
        optimizer_state = solver.init(optimizer, control)

        return environment_state, optimizer_state

    @eqx.jit_filtered
    def _step(
        optimizer_state: optax.OptState,
        optimizer: optax.GradientTransformation,
        environment_state: environments.EnvironmentState,
        environment: environments.AbstractEnvironment,
        rewards: Callable[[Array], ArrayLike],
        _constraints: List[constraints.AbstractConstraint],
        control: controls.AbstractControl,
        solver: solvers.AbstractSolver,
    ) -> Tuple[ArrayLike, controls.AbstractControl]:
        return solver.step(
            optimizer_state,
            optimizer,
            environment_state,
            environment,
            rewards,
            _constraints,
            control,
        )

    optimizer = optax.adam(learning_rate=1e-2)
    environment_state, optimizer_state = _init(environment, solver, control, optimizer)

    pbar = trange(num_steps)
    for _ in pbar:
        reward, control = _step(
            optimizer_state,
            optimizer,
            environment_state,
            environment,
            rewards,
            _constraints,
            control,
            solver,
        )

        pbar.set_postfix({"reward": reward})

    return reward, control
