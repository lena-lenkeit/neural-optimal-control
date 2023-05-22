import abc
from functools import partial
from typing import Any, Callable, List, Optional, Tuple

import diffrax
import equinox as eqx
import evosax
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, ArrayLike, PyTree

import optimal_control.constraints as constraints
import optimal_control.controls as controls
import optimal_control.environments as environments
import optimal_control.solvers as solvers
from optimal_control.utils import default, exists


class ESSolverState(solvers.SolverState):
    strategy_state: evosax.EvoState


class ESSolver(solvers.AbstractSolver):
    strategy: evosax.Strategy
    strategy_params: evosax.EvoParams
    parameter_reshaper: evosax.ParameterReshaper
    fitness_shaper: evosax.FitnessShaper
    num_control_points: int

    def init(
        self, control: controls.AbstractControl, key: jax.random.KeyArray
    ) -> ESSolverState:
        control_params = eqx.filter(control, eqx.is_array)
        strategy_params = default(self.strategy_params, self.strategy.default_params)
        strategy_state = self.strategy.initialize(
            rng=key,
            params=strategy_params,
            init_mean=self.parameter_reshaper.flatten_single(control_params),
        )

        return ESSolverState(strategy_state=strategy_state)

    def step(
        self,
        state: ESSolverState,
        environment: environments.AbstractEnvironment,
        environment_state: environments.EnvironmentState,
        reward_fn: Callable[[PyTree], float],
        constraint_chain: List[constraints.AbstractConstraint],
        control: controls.AbstractControl,
        key: jax.random.KeyArray,
    ) -> Tuple[ESSolverState, controls.AbstractControl, float]:
        strategy_key, fitness_key = jax.random.split(key)

        # Get candidates
        control_params_flat, strategy_state = self.strategy.ask(
            strategy_key, strategy_state, self.strategy_params
        )

        # Build vectorized controls
        control_params_pytree = self.parameter_reshaper.reshape(control_params_flat)
        control_static_pytree = eqx.filter(control, eqx.is_array, inverse=True)
        control_population = eqx.combine(control_params_pytree, control_static_pytree)

        # Make fitness function
        fitness_fn = eqx.filter_vmap(
            partial(
                solvers.evaluate_reward,
                constraint_chain=constraint_chain,
                environment=environment,
                environment_state=environment_state,
                reward_fn=reward_fn,
                num_control_points=self.num_control_points,
                key=fitness_key,
            )
        )

        # Evaluate candidates
        fitness = fitness_fn(control_population)
        fitness = self.fitness_shaper.apply(control_params_flat, fitness).astype(
            jnp.float32
        )

        strategy_state = self.strategy.tell(
            control_params_flat, fitness, strategy_state, self.strategy_params
        )

        # Extract best control
        best_control_params_flat = strategy_state.best_member
        best_control_params_pytree = self.parameter_reshaper.reshape(
            best_control_params_flat
        )
        best_control = eqx.combine(best_control_params_pytree, control_static_pytree)

        return (
            ESSolverState(strategy_state=strategy_state),
            best_control,
            strategy_state.best_fitness,
        )
