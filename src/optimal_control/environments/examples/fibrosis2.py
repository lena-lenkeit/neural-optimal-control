# A less stiff and more physiologically plausible variant of the fibrosis model

from functools import partial
from typing import Callable, Tuple

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree, Scalar

import optimal_control
import optimal_control.controls as controls
import optimal_control.environments as environments


def _fibrosis_ode(t: Scalar, y: Array, u: Array, m: Array) -> Array:
    c_pdgf_ab, c_csf1_ab = u
    macrophage_influx = m

    # Constants
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
    k[12] = macrophage_influx * 140 * 1440  # + 0.01 * 1440  # 120 inflammation pulse
    k[13] = 1e6

    # PDGF antibody
    k_pdfg_ab = 1 * 1440  # 1 / (min * molecule)
    pdgf_ab_deg = -k_pdfg_ab * y[3] * jnp.clip(c_pdgf_ab, a_min=1e-4)

    # CSF1 antibody
    k_csf1_ab = 1 * 1440  # 1 / (min * molecule)
    csf1_ab_deg = -k_csf1_ab * y[2] * jnp.clip(c_csf1_ab, a_min=1e-4)

    # ODE
    dy = [None] * 4

    dy[0] = y[0] * (
        k[0] * y[3] / (k[10] + y[3]) * (1 - y[0] / k[3]) - k[2]
    )  # + 0.01 * 1440  # Fibrobasts
    dy[1] = y[1] * (k[1] * y[2] / (k[11] + y[2]) - k[2]) + k[12]  # Mph
    dy[2] = (
        csf1_ab_deg + k[6] * y[0] - k[8] * y[1] * y[2] / (k[11] + y[2]) - k[4] * y[2]
    )  # CSF
    dy[3] = (
        pdgf_ab_deg
        + k[7] * y[1]
        + k[5] * y[0]
        - k[9] * y[0] * y[3] / (k[10] + y[3])
        - k[4] * y[3]
    )  # PDGF

    return jnp.stack(dy, axis=-1)


def fibrosis_ode(
    t: Scalar, y: Array, u: Array, args: Callable[[Scalar], Scalar]
) -> Array:
    m = args
    return _fibrosis_ode(t, y, u, m(t=t)[0])


def _fibrosis_reward(t: Scalar, fy: PyTree, gy: PyTree, u: PyTree, args: PyTree):
    fibrosis_penalty = jnp.sum(jnp.log(jnp.clip(fy[..., :2], a_min=1e2)), axis=-1)

    return -jnp.atleast_1d(fibrosis_penalty)


def influx_fn(t: Scalar) -> Array:
    return jnp.where(t < 4.0, jnp.ones(1), jnp.zeros(1))


def zero_fn(*args, **kwargs) -> Array:
    return jnp.zeros(1)


class FibrosisState(environments.EnvironmentState):
    y0: Array


class FibrosisEnvironment(environments.AbstractEnvironment):
    def init(self) -> FibrosisState:
        return FibrosisState(
            self._integrate(
                t0=0.0,
                t1=300.0,
                y0=jnp.asarray([1e8, 1e8, 1e8, 1e8]),
                control=controls.LambdaControl(lambda _: jnp.zeros(2)),
                inflammation_pulse=False,
                throw=True,
                stepsize_controller=diffrax.PIDController(
                    rtol=1e-6, atol=1e-6, pcoeff=1.0, icoeff=1.0
                ),
            ).ys[-1, :4]
        )

    def _integrate(
        self,
        t0: float,
        t1: float,
        y0: Array,
        control: controls.AbstractControl,
        inflammation_pulse: bool,
        saveat: diffrax.SaveAt = diffrax.SaveAt(t1=True),
        max_steps: int = 1000,
        throw: bool = False,
        stepsize_controller: diffrax.AbstractStepSizeController = diffrax.PIDController(
            rtol=1e-6, atol=1e-6, pcoeff=1.0, icoeff=1.0
        ),
        dt0: float = 0.1,
    ) -> diffrax.Solution:
        ode = fibrosis_ode
        ode = optimal_control.with_extra_term(
            ode, g=_fibrosis_reward, num_g_states=1, g0=jnp.zeros(1)
        )
        ode = optimal_control.with_control(ode, time=True)

        terms = diffrax.ODETerm(ode)
        solver = diffrax.Kvaerno5()

        macrophage_influx_control = controls.LambdaControl(
            (lambda state: influx_fn(state["t"])) if inflammation_pulse else zero_fn
        )

        solution = diffrax.diffeqsolve(
            terms=terms,
            solver=solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=ode._modify_initial_state(control, t0, y0),
            # y0=y0,
            args=(control, macrophage_influx_control),
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
            adjoint=diffrax.RecursiveCheckpointAdjoint(checkpoints=max_steps),
            throw=throw,
        )

        return solution

    def integrate(
        self,
        control: controls.AbstractControl,
        state: FibrosisState,
        key: jax.random.KeyArray,
        *,
        saveat: diffrax.SaveAt = diffrax.SaveAt(t1=True),
        # saveat: diffrax.SaveAt = diffrax.SaveAt(ts=jnp.linspace(0.0, 200.0, 201)),
        max_steps: int = 10000,
        throw: bool = True,
        stepsize_controller: diffrax.AbstractStepSizeController = diffrax.PIDController(
            rtol=1e-4, atol=1e-4, pcoeff=1.0, icoeff=1.0, dtmax=1.0
        ),
        dt0: float = 0.1,
    ) -> diffrax.Solution:
        solution = self._integrate(
            t0=0.0,
            t1=200.0,
            y0=state.y0,
            control=control,
            inflammation_pulse=False,
            saveat=saveat,
            max_steps=max_steps,
            throw=throw,
            stepsize_controller=stepsize_controller,
            dt0=dt0,
        )

        return solution
