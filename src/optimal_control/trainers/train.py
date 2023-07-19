from functools import partial
from typing import Any, Callable, List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax_tqdm as jtq
import optax
from jax import lax
from jaxtyping import Array, ArrayLike, PyTree, Scalar
from tqdm.auto import tqdm as tq
from tqdm.auto import trange

import optimal_control.constraints as constraints
import optimal_control.controls as controls
import optimal_control.environments as environments
import optimal_control.solvers as solvers
from optimal_control.utils import exists


class TrainState(eqx.Module):
    control: controls.AbstractControl
    solver_state: solvers.SolverState
    reward: float
    key: jax.random.KeyArray


def init_state(
    environment: environments.AbstractEnvironment,
    solver: solvers.AbstractSolver,
    control: controls.AbstractControl,
) -> Tuple[environments.EnvironmentState, solvers.SolverState]:
    environment_state = environment.init()
    solver_state = solver.init(control)

    return environment_state, solver_state


def step_state(
    i: int,
    train_state_jaxtypes: TrainState,
    train_state_pytypes: TrainState,
    solver: solvers.AbstractSolver,
    environment: environments.AbstractEnvironment,
    environment_state: environments.EnvironmentState,
    reward_fn: Callable[[PyTree], float],
    constraint_chain: constraints.ConstraintChain,
    integrate_kwargs: dict,
) -> TrainState:
    train_state: TrainState = eqx.combine(train_state_jaxtypes, train_state_pytypes)

    key, subkey = jax.random.split(train_state.key)
    solver_state, control, reward = solver.step(
        state=train_state.solver_state,
        environment=environment,
        environment_state=environment_state,
        reward_fn=reward_fn,
        constraint_chain=constraint_chain,
        control=train_state.control,
        key=subkey,
        integrate_kwargs=integrate_kwargs,
    )

    train_state = TrainState(
        control=control, solver_state=solver_state, reward=reward, key=key
    )

    train_state_jaxtypes = eqx.filter(train_state, eqx.is_array)
    return train_state_jaxtypes


def solve_optimal_control_problem(
    num_train_steps: int,
    environment: environments.AbstractEnvironment,
    reward_fn: Callable[[PyTree], float],
    constraint_chain: constraints.ConstraintChain,
    solver: solvers.AbstractSolver,
    control: controls.AbstractControl,
    key: jax.random.KeyArray,
    pbar_interval: Optional[int] = None,
    integrate_kwargs: dict = {},
) -> Tuple[float, controls.AbstractControl]:
    # Initialize states
    environment_state, solver_state = init_state(
        environment=environment, solver=solver, control=control
    )

    train_state = TrainState(
        control=control, solver_state=solver_state, reward=jnp.float_(0.0), key=key
    )

    train_state_jaxtypes, train_state_pytypes = eqx.partition(train_state, eqx.is_array)

    # Build loop step function
    step_fn = partial(
        step_state,
        train_state_pytypes=train_state_pytypes,
        solver=solver,
        environment=environment,
        environment_state=environment_state,
        reward_fn=reward_fn,
        constraint_chain=constraint_chain,
        integrate_kwargs=integrate_kwargs,
    )

    if exists(pbar_interval):
        step_fn = jtq.loop_tqdm(n=num_train_steps, print_rate=pbar_interval)(step_fn)

    # Run training loop
    train_state_jaxtypes = lax.fori_loop(
        lower=0,
        upper=num_train_steps,
        body_fun=step_fn,
        init_val=train_state_jaxtypes,
    )

    train_state = eqx.combine(train_state_jaxtypes, train_state_pytypes)
    return train_state.reward, train_state.control
