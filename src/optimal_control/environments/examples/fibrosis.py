from functools import partial
from typing import Callable, Tuple

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import optimal_control.controls as controls
import optimal_control.environments as environments


# @partial(jax.jit, static_argnums=(3, 4))
def fibrosis_ode(
    t: ArrayLike, y: Array, args: Tuple[controls.AbstractControl, bool]
) -> Array:
    u, inflammation_pulse = args

    k = [None] * 14

    k[0] = 0.9  # proliferation rates: lambda1=0.9/day,
    k[1] = 0.8  # lambda2=0.8/day
    k[2] = 0.3  # mu_1, mu_2, death rates: 0.3/day
    k[3] = 1e6  # carrying capacity: 10^6 cells
    k[4] = 2  # growth factor degradation: gamma=2/day
    k[5] = (
        240 * 1440
    )  # growth factor secretion rates: beta3=240 molecules/cell/min  ---- beta_3
    k[6] = (
        470 * 1440
    )  # beta1=470 molecules/cell/min                                ---- beta_1
    k[7] = (
        70 * 1440
    )  # beta2=70 molecules/cell/min                                 ---- beta_2
    k[8] = (
        940 * 1440
    )  # alpha1=940 molecules/cell/min, endocytosis rate CSF1       ---- alpha_1
    k[9] = (
        510 * 1440
    )  # alpha2=510 molecules/cell/min, endocytosis rate PDGF     ---- alpha_2
    k[10] = 6e8  # #binding affinities: k1=6x10^8 molecules (PDGF)     ---- k_1
    k[11] = 6e8  # k2=6x10^8 (CSF)                                   ---- k_2
    k[12] = (
        jnp.where(t < 4, 140 * 1440, 0) if inflammation_pulse else 0
    )  # 120 inflammation pulse
    k[13] = 1e6

    # Control
    """vinterp = jax.vmap(jnp.interp, in_axes=(None, None, -1), out_axes=-1)

    u_at_t = vinterp(
        t,
        jnp.linspace(0.0, 100.0, 100 + 1),
        jnp.concatenate((jnp.zeros_like(u[:1]), u), axis=0),
    )"""

    # u = eqx.combine(u_params, u_static)
    u_at_t = u(t)

    # PDGF antibody
    k_pdfg_ab = 1 * 1440  # 1 / (min * molecule)
    pdgf_ab_deg = -k_pdfg_ab * y[3] * u_at_t[0]

    # CSF1 antibody
    k_csf1_ab = 1 * 1440  # 1 / (min * molecule)
    csf1_ab_deg = -k_csf1_ab * y[2] * u_at_t[1]

    # Cytostatic drug
    # k[0] = 0.9 * (1 - u_at_t[1] / (u_at_t[1] + 1.0))

    dx = [None] * 4

    dx[0] = y[0] * (
        k[0] * y[3] / (k[10] + y[3]) * (1 - y[0] / k[3]) - k[2]
    )  # Fibrobasts
    dx[1] = y[1] * (k[1] * y[2] / (k[11] + y[2]) - k[2]) + k[12]  # Mph
    dx[2] = (
        csf1_ab_deg + k[6] * y[0] - k[8] * y[1] * y[2] / (k[11] + y[2]) - k[4] * y[2]
    )  # CSF
    dx[3] = (
        pdgf_ab_deg
        + k[7] * y[1]
        + k[5] * y[0]
        - k[9] * y[0] * y[3] / (k[10] + y[3])
        - k[4] * y[3]
    )  # PDGF

    # cells = jnp.stack(y[:2], axis=-1)
    # cells = jnp.clip(cells, a_min=1e2, a_max=None)
    # cells_fibrosis = jnp.asarray([4.84813927e05, 6.45000129e05])

    # dx[4] = jnp.mean(jnp.log(cells_fibrosis) - jnp.log(cells))
    # dx[4] = -jnp.mean(jnp.log(jnp.clip(cells, a_min=1e2, a_max=None)))

    return jnp.stack(dx, axis=-1)


class FibrosisState(environments.EnvironmentState):
    y0: Array


class FibrosisEnvironment(environments.AbstractEnvironment):
    def init(self) -> FibrosisState:
        return FibrosisState(
            self._integrate(
                0.0,
                300.0,
                jnp.asarray([1.0, 1.0, 0.0, 0.0]),
                controls.LambdaControl(lambda _: jnp.zeros(2)),
                True,
                diffrax.SaveAt(t1=True),
                max_steps=100000,
                throw=True,
            ).ys[-1, :4]
        )

    def _integrate(
        self,
        t0: float,
        t1: float,
        y0: Array,
        control: controls.AbstractControl,
        inflammation_pulse: bool,
        saveat: diffrax.SaveAt,
        early_stopping: bool = False,
        max_steps: int = 1000,
        throw: bool = False,
    ) -> diffrax.Solution:
        terms = diffrax.ODETerm(fibrosis_ode)
        # solver = diffrax.ImplicitEuler(
        #    nonlinear_solver=diffrax.NewtonNonlinearSolver(rtol=1e-5, atol=1e-5)
        # )
        solver = diffrax.Kvaerno5()
        stepsize_controller = diffrax.PIDController(
            rtol=1e-5, atol=1e-5, pcoeff=0.3, icoeff=0.3
        )

        # def cond_fn(state, **kwargs) -> bool:
        #    return (state.y[0] < 1e2) & (state.y[1] < 1e2)

        # event = diffrax.DiscreteTerminatingEvent(cond_fn)

        sol = diffrax.diffeqsolve(
            terms=terms,
            solver=solver,
            t0=t0,
            t1=t1,
            dt0=0.1,
            y0=jnp.concatenate((y0, jnp.zeros_like(y0[:1])), axis=0),
            # y0=y0,
            args=(control, inflammation_pulse),
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
            adjoint=diffrax.RecursiveCheckpointAdjoint(checkpoints=max_steps),
            throw=throw,
            # discrete_terminating_event=event if early_stopping else None,
        )

        return sol

    def integrate(
        self,
        control: controls.AbstractControl,
        state: FibrosisState,
        key: jax.random.KeyArray,
    ) -> Array:
        sol = self._integrate(
            0.0,
            200.0,
            state.y0,
            control,
            False,
            # diffrax.SaveAt(ts=jnp.linspace(0.0, 200.0, 201)),
            diffrax.SaveAt(t1=True),
            True,
            max_steps=1000,
            throw=False,
        )

        return sol.ys  # , sol.ts
