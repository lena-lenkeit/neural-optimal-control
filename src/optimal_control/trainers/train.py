from typing import Any, Callable, List, Tuple

import equinox as eqx
from jaxtyping import Array, ArrayLike
from tqdm.auto import tqdm as tq
from tqdm.auto import trange

import optimal_control.environments as environments
import optimal_control.solvers as solvers


def solve_optimal_control_problem(
    environment: environments.AbstractEnvironment,
    rewards: Callable[[Array], ArrayLike],
    constraints: List[solvers.AbstractConstraint],
    solver: solvers.AbstractSolver,
    control: solvers.AbstractControl,
    num_steps: int,
):
    def _init(
        environment: environments.AbstractEnvironment,
        solver: solvers.AbstractSolver,
        control: solvers.AbstractControl,
    ):
        ...

    @eqx.jit_filtered
    def _step(
        environment: environments.AbstractEnvironment,
        rewards: Callable[[Array], ArrayLike],
        constraints: List[solvers.AbstractConstraint],
        solver: solvers.AbstractSolver,
        control: solvers.AbstractControl,
    ) -> Tuple[ArrayLike, solver.AbstractControl]:
        ...

    for _ in trange(num_steps):
        ...
