from functools import partial
from typing import Any, Callable, List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax_tqdm as jtq
import optax
from jax import lax
from jaxtyping import Array, ArrayLike
from tqdm.auto import tqdm as tq
from tqdm.auto import trange

import optimal_control.constraints as constraints
import optimal_control.controls as controls
import optimal_control.environments as environments
import optimal_control.solvers as solvers


class TrainState(eqx.Module):
    optimizer_state: optax.OptState
    optimizer: optax.GradientTransformation
    environment_state: environments.EnvironmentState
    environment: environments.AbstractEnvironment
    rewards: Callable[[Array], ArrayLike]
    reward: ArrayLike
    _constraints: List[constraints.AbstractConstraint]
    control: controls.AbstractControl
    solver: solvers.AbstractSolver
    key: jax.random.KeyArray


def solve_optimal_control_problem(
    environment: environments.AbstractEnvironment,
    rewards: Callable[[Array], ArrayLike],
    _constraints: List[constraints.AbstractConstraint],
    solver: solvers.AbstractSolver,
    control: controls.AbstractControl,
    num_steps: int,
    key: jax.random.KeyArray,
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

    def _step(
        i: int, train_state_jaxtypes: TrainState, train_state_pytypes: TrainState
    ) -> TrainState:
        train_state = eqx.combine(train_state_jaxtypes, train_state_pytypes)

        key, subkey = jax.random.split(train_state.key)
        reward, control, optimizer_state = solver.step(
            train_state.optimizer_state,
            train_state.optimizer,
            train_state.environment_state,
            train_state.environment,
            train_state.rewards,
            train_state._constraints,
            train_state.control,
            subkey,
        )

        train_state = TrainState(
            optimizer_state=optimizer_state,
            optimizer=train_state.optimizer,
            environment_state=train_state.environment_state,
            environment=train_state.environment,
            rewards=train_state.rewards,
            reward=reward,
            _constraints=train_state._constraints,
            control=control,
            solver=train_state.solver,
            key=key,
        )

        train_state_jaxtypes, train_state_pytypes = eqx.partition(
            train_state, eqx.is_array
        )
        # print(train_state_jaxtypes, train_state_pytypes)
        return train_state_jaxtypes

    optimizer = optax.adam(learning_rate=1e-1)
    environment_state, optimizer_state = _init(environment, solver, control, optimizer)

    train_state = TrainState(
        optimizer_state=optimizer_state,
        optimizer=optimizer,
        environment_state=environment_state,
        environment=environment,
        rewards=rewards,
        reward=jnp.float64(0.0),
        _constraints=_constraints,
        control=control,
        solver=solver,
        key=key,
    )

    train_state_jaxtypes, train_state_pytypes = eqx.partition(train_state, eqx.is_array)
    # print(train_state_jaxtypes, train_state_pytypes)
    train_state_jaxtypes = lax.fori_loop(
        0,
        num_steps,
        jtq.loop_tqdm(num_steps, print_rate=10)(
            partial(_step, train_state_pytypes=train_state_pytypes)
        ),
        train_state_jaxtypes,
    )
    train_state = eqx.combine(train_state_jaxtypes, train_state_pytypes)

    return train_state.reward, train_state.control

    """@eqx.filter_jit
    def _step(
        optimizer_state: optax.OptState,
        optimizer: optax.GradientTransformation,
        environment_state: environments.EnvironmentState,
        environment: environments.AbstractEnvironment,
        rewards: Callable[[Array], ArrayLike],
        _constraints: List[constraints.AbstractConstraint],
        control: controls.AbstractControl,
        solver: solvers.AbstractSolver,
    ) -> Tuple[ArrayLike, controls.AbstractControl, optax.OptState]:
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
        reward, control, optimizer_state = _step(
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
    """
