from functools import partial
from typing import Callable

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import optimal_control.controls as controls
import optimal_control.environments as environments


@partial(jax.jit, static_argnums=(3,))
def zero_ode(
    x: Array,
    t: ArrayLike,
    u_params: controls.AbstractControl,
    u_static: controls.AbstractControl,
) -> Array:
    u = eqx.combine(u_params, u_static)
    return u(t)


class ZeroState(environments.EnvironmentState):
    state: Array


class ZeroEnvironment(environments.AbstractEnvironment):
    def init(self) -> ZeroState:
        return ZeroState(jnp.zeros(1))

    def integrate(self, control: controls.AbstractControl, state: ZeroState) -> Array:
        # return control(jnp.linspace(0.0, 100.0, 101)) # Works

        terms = diffrax.ODETerm(lambda t, y, args: zero_ode(y, t, *args))
        solver = diffrax.Dopri5()
        # stepsize_controller = diffrax.PIDController(rtol=1e-7, atol=0.0)

        sol = diffrax.diffeqsolve(
            terms,
            solver,
            0.0,
            200.0,
            0.1,
            jnp.zeros(2),
            args=eqx.partition(control, eqx.is_array),
            saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, 200.0, 201)),
            # stepsize_controller=stepsize_controller,
            max_steps=100000,
        )

        return sol.ys
