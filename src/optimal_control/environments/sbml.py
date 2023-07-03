from typing import Callable, Union

import diffrax
import jax
import jax.numpy as jnp
import libsbml
from jaxtyping import PRNGKeyArray, PyTree, Scalar

import optimal_control.controls as controls
import optimal_control.environments as environments
import optimal_control.sbml as sbmlutils
from optimal_control.environments.base import EnvironmentState


class SBMLEnvironmentState(environments.EnvironmentState):
    ode: Callable[[Scalar, PyTree, PyTree], PyTree]
    y0: PyTree


class SBMLEnvironment(environments.AbstractEnvironment):
    model_or_filepath: Union[libsbml.Model, str]
    control_input_fn: Callable[[Scalar, PyTree], PyTree]
    control_output_fn: Callable[[PyTree], PyTree]
    ode_mod_fn: Callable[
        [Callable[[Scalar, PyTree, PyTree, PyTree], PyTree]],
        Callable[[Scalar, PyTree, PyTree], PyTree],
    ]
    t0: float
    t1: float
    dt0: float
    solver: diffrax.AbstractSolver = diffrax.Kvaerno5()
    stepsize_controller: diffrax.AbstractStepSizeController = diffrax.PIDController(
        rtol=1e-5, atol=1e-5
    )
    saveat: diffrax.SaveAt = diffrax.SaveAt(dense=True)

    def init(self) -> SBMLEnvironmentState:
        if isinstance(self.model_or_filepath, libsbml.Model):
            model = self.model_or_filepath
        else:
            model: libsbml.Model = libsbml.readSBMLFromFile(
                self.model_or_filepath
            ).getModel()

        return SBMLEnvironmentState(
            ode=self.ode_mod_fn(sbmlutils.model_to_lambda(model)),
            y0=sbmlutils.species_to_dict(model.getListOfSpecies()),
        )

    def integrate(
        self,
        control: controls.AbstractControl,
        state: SBMLEnvironmentState,
        key: PRNGKeyArray,
    ) -> PyTree:
        def ode_fn(t: Scalar, y: PyTree, args: PyTree) -> PyTree:
            control, overrides = args

            control_inputs = self.control_input_fn(t, y)
            control_values = control(control_inputs)
            control_outputs = self.control_output_fn(control_values)

            dy_dt = state.ode(t, y, (control_outputs, overrides))
            return dy_dt

        terms = diffrax.ODETerm(ode_fn)
        solution = diffrax.diffeqsolve(
            terms=terms,
            solver=self.solver,
            t0=self.t0,
            t1=self.t1,
            dt0=self.dt0,
            y0=state.y0,
            args=control,
            saveat=self.saveat,
            stepsize_controller=self.stepsize_controller,
            max_steps=10000,
        )

        return solution
