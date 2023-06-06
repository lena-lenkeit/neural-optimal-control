from functools import partial
from typing import Callable, Tuple

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Scalar

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
    k[12] = 0.01 * 1440 + macrophage_influx * 140 * 1440  # 120 inflammation pulse
    k[13] = 1e6

    # PDGF antibody
    k_pdfg_ab = 1 * 1440  # 1 / (min * molecule)
    pdgf_ab_deg = -k_pdfg_ab * y[3] * c_pdgf_ab

    # CSF1 antibody
    k_csf1_ab = 1 * 1440  # 1 / (min * molecule)
    csf1_ab_deg = -k_csf1_ab * y[2] * c_csf1_ab

    # ODE
    dy = [None] * 4

    dy[0] = (
        y[0] * (k[0] * y[3] / (k[10] + y[3]) * (1 - y[0] / k[3]) - k[2]) + 0.01 * 1440
    )  # Fibrobasts
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
    return _fibrosis_ode(t, y, u, m(t)[0])


class FibrosisState(environments.EnvironmentState):
    ...


class FibrosisEnvironment(environments.AbstractEnvironment):
    def init(self) -> FibrosisState:
        return FibrosisState()

    def _integrate(
        self,
        t0: float,
        t1: float,
        y0: Array,
        control: controls.AbstractControl,
        influx: controls.AbstractControl,
        saveat: diffrax.SaveAt,
        max_steps: int = 1000,
        throw: bool = False,
    ) -> diffrax.Solution:
        ode = optimal_control.with_control(fibrosis_ode)
        terms = diffrax.ODETerm(ode)
        solver = diffrax.Kvaerno3()
        stepsize_controller = diffrax.PIDController(
            rtol=1e-4,
            atol=1e-4,
            pcoeff=0.1,
            icoeff=0.3,
            jump_ts=jnp.linspace(t0, t1, jnp.int_((t1 - t0) / 4), endpoint=False) + 4,
        )

        sol = diffrax.diffeqsolve(
            terms=terms,
            solver=solver,
            t0=t0,
            t1=t1,
            dt0=0.1,
            y0=y0,
            args=(control, influx),
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
            adjoint=diffrax.RecursiveCheckpointAdjoint(checkpoints=max_steps),
            throw=throw,
        )

        return sol

    def _sample_influx_sequence_twostate(
        self, t0: float, t1: float, dt: float, p01: float, p10: float, key: PRNGKeyArray
    ) -> Array:
        # Models the influx as a two-state system with transition probabilities
        # 0 -> 1 and 1 -> 0

        def f(state: int, key: PRNGKeyArray) -> int:
            w = jax.random.uniform(key)
            p = jnp.where(state == 1, p10, p01)

            next_state = jnp.where(w < p, 1 - state, state)
            return next_state, next_state

        seq_len = jnp.int_(jnp.ceil((t1 - t0) // dt))
        _, influx_seq = jax.lax.scan(
            f, init=jnp.int_(0), xs=jax.random.split(key, num=seq_len)
        )

        return influx_seq

    def _sample_influx_sequence_random(
        self, t0: float, t1: float, dt: float, p1: float, key: PRNGKeyArray
    ) -> Array:
        # Models the influx as as having a random chance p of occuring at each
        # interval dt

        seq_len = jnp.int_(jnp.ceil((t1 - t0) // dt))

        return jax.random.bernoulli(key, p=p1, shape=(seq_len,))

    def integrate(
        self,
        control: controls.AbstractControl,
        state: FibrosisState,
        key: jax.random.KeyArray,
    ) -> Array:
        t0 = 0.0
        t1 = 1000.0

        # influx_sequence = self._sample_influx_sequence(
        #    t0=t0, t1=t1, dt=1.0, p01=0.01, p10=0.25, key=key
        # )

        influx_sequence = self._sample_influx_sequence_random(
            t0=t0, t1=t1, dt=4.0, p1=0.04, key=key
        )

        # influx_sequence = jnp.zeros_like(influx_sequence)

        influx_control = controls.InterpolationControl(
            channels=1,
            steps=len(influx_sequence),
            t_start=t0,
            t_end=t1,
            method="step",
            control=influx_sequence.reshape(-1, 1),
        )

        sol = self._integrate(
            t0=t0,
            t1=t1,
            y0=jnp.ones(4),
            control=control,
            influx=influx_control,
            saveat=diffrax.SaveAt(ts=jnp.linspace(t0, t1, 1001)),
            max_steps=10000,
        )

        return sol
