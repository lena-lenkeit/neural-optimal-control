from typing import List

from optimal_control.constraints import AbstractConstraint
from optimal_control.solvers import AbstractSolver


def solve_optimal_control_problem(
    environment, rewards, constraints: List[AbstractConstraint], solver: AbstractSolver
):
    pass
