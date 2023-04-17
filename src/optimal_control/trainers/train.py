from typing import Any, Callable, List

from jaxtyping import Array, ArrayLike
from tqdm.auto import tqdm as tq

from optimal_control.constraints import AbstractConstraint
from optimal_control.solvers import AbstractSolver


def solve_optimal_control_problem(
    environment: Callable[[Array, ArrayLike, Any], Array],
    rewards: Callable[[Array], ArrayLike],
    constraints: List[AbstractConstraint],
    solver: AbstractSolver,
):
    pass
