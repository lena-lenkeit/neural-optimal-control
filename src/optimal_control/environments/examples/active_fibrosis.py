from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PRNGKeyArray, PyTree, Scalar

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


def _fibrosis_reward(t: Scalar, fy: PyTree, gy: PyTree, u: PyTree, args: PyTree):
    fibrosis_penalty = jnp.sum(jnp.log(fy[..., :2]), axis=-1)
    antibody_penalty = jnp.sum(u, axis=-1) * 100000.0

    return -jnp.atleast_1d(fibrosis_penalty + antibody_penalty)


def fibrosis_ode(
    t: Scalar, y: Array, u: Array, args: Callable[[Scalar], Scalar]
) -> Array:
    m = args
    return _fibrosis_ode(t, y, u, m(t)[0])


class FibrosisState(environments.EnvironmentState):
    y0: Array


class FibrosisEnvironment(environments.AbstractEnvironment):
    def init(self) -> FibrosisState:
        solution = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(
                lambda t, y, args: _fibrosis_ode(t, y, jnp.zeros(2), jnp.float_(0.0))
            ),
            solver=diffrax.Kvaerno5(),
            t0=0.0,
            t1=300.0,
            dt0=0.1,
            y0=jnp.ones(4),
            stepsize_controller=diffrax.PIDController(
                rtol=1e-7, atol=1e-7, pcoeff=0.1, icoeff=0.3
            ),
        )

        return FibrosisState(y0=solution.ys[-1])

    def _integrate_waitingtimes(
        self,
        t0: float,
        t1: float,
        y0: Array,
        control: controls.AbstractControl,
        influx_events: Array,
        saveat: diffrax.SaveAt,
        max_steps: int = 1000,
        dense: bool = False,
        throw: bool = False,
    ) -> Array:
        ode = _fibrosis_ode
        ode = optimal_control.with_extra_term(
            ode, g=_fibrosis_reward, num_g_states=1, g0=jnp.zeros(1)
        )
        # ode = optimal_control.with_cde_rnn_control(ode, num_latents=4)
        ode = optimal_control.with_control(ode)

        terms = diffrax.ODETerm(ode)
        solver = diffrax.Kvaerno5()
        stepsize_controller = diffrax.PIDController(
            rtol=1e-2, atol=1e-2, pcoeff=0.1, icoeff=0.3
        )

        y0 = ode._modify_initial_state(control, t0, y0)
        if dense:
            max_sub_steps = max_steps // len(influx_events)
        else:
            max_sub_steps = max_steps

        class Carry(eqx.Module):
            t0: float
            y0: PyTree
            event_idx: int
            solver_state: PyTree
            stepsize_controller_state: PyTree
            results: diffrax.RESULTS
            dense_idx: Optional[int]
            interpolation: Optional[diffrax.DenseInterpolation]

        def step_fn(
            carry: Carry, first_step: bool = False, dense: bool = dense
        ) -> Carry:
            influx_t1 = jax.lax.cond(
                carry.event_idx < len(influx_events),
                lambda: jnp.minimum(influx_events[carry.event_idx], t1),
                lambda: t1,
            )

            if first_step:
                y0 = carry.y0
            else:
                y0 = carry.y0.at[..., 1].add(1e6)

            solution = diffrax.diffeqsolve(
                terms=terms,
                solver=solver,
                t0=carry.t0,
                t1=influx_t1,
                dt0=0.1,
                y0=y0,
                args=(control, jnp.float_(0.0)),
                saveat=diffrax.SaveAt(
                    t1=True, dense=dense, solver_state=True, controller_state=True
                ),
                stepsize_controller=stepsize_controller,
                max_steps=max_sub_steps,
                throw=throw,
                solver_state=carry.solver_state,
                controller_state=carry.stepsize_controller_state,
                made_jump=True,
            )

            if dense:
                if first_step:
                    ts = jnp.full(
                        (max_steps + 1,), jnp.nan, solution.interpolation.ts.dtype
                    )
                    infos = jax.tree_map(
                        lambda info: jnp.full(
                            (max_steps,) + info.shape[1:], jnp.nan, info.dtype
                        ),
                        solution.interpolation.infos,
                    )
                    dense_idx = 0
                else:
                    ts = carry.interpolation.ts
                    infos = carry.interpolation.infos
                    dense_idx = carry.dense_idx

                next_dense_idx = dense_idx + solution.interpolation.ts_size - 1
                next_dense_idx = eqx.internal.error_if(
                    next_dense_idx,
                    next_dense_idx >= max_steps,
                    "Exceeded the maximum number of steps!",
                )

                next_ts = jax.lax.dynamic_update_slice_in_dim(
                    ts, solution.interpolation.ts, dense_idx, axis=0
                )
                next_infos = jax.tree_map(
                    lambda operand, update: jax.lax.dynamic_update_slice_in_dim(
                        operand, update, dense_idx, axis=0
                    ),
                    infos,
                    solution.interpolation.infos,
                )

                next_interpolation = diffrax.DenseInterpolation(
                    ts=next_ts,
                    ts_size=next_dense_idx + 1,
                    infos=next_infos,
                    interpolation_cls=solution.interpolation.interpolation_cls,
                    direction=solution.interpolation.direction,
                    t0_if_trivial=solution.interpolation.t0_if_trivial,
                    y0_if_trivial=solution.interpolation.y0_if_trivial,
                )
            else:
                next_dense_idx = None
                next_interpolation = None

            next_carry = Carry(
                t0=solution.t1,
                y0=solution.ys[-1],
                event_idx=carry.event_idx + 1,
                solver_state=solution.solver_state,
                stepsize_controller_state=solution.controller_state,
                results=solution.result,
                dense_idx=next_dense_idx,
                interpolation=next_interpolation,
            )

            return next_carry

        def cond_fn(carry: Carry) -> bool:
            return diffrax.is_successful(carry.results) & (carry.t0 < t1)

        # Do first step to populate carry
        init_carry = Carry(
            t0=0.0,
            y0=y0,
            event_idx=0,
            solver_state=None,
            stepsize_controller_state=None,
            results=diffrax.RESULTS.successful,
            dense_idx=None,
            interpolation=None,
        )

        init_carry = step_fn(init_carry, first_step=True)

        # Loop over events
        final_carry = eqx.internal.while_loop(
            cond_fn,
            step_fn,
            init_val=init_carry,
            kind="checkpointed",
            max_steps=len(influx_events),
            checkpoints=len(influx_events),
        )

        # y1 = jnp.where(
        #    diffrax.is_successful(final_carry.results),
        #    final_carry.y0,
        #    jax.lax.stop_gradient(final_carry.y0),
        # )

        # y1 = jax.lax.cond(
        #    diffrax.is_successful(final_carry.results)
        #    & jnp.all(final_carry.y0 != jnp.nan),
        #    lambda y1: y1,
        #    lambda y1: jnp.zeros_like(y1),
        #    final_carry.y0,
        # )

        y1 = final_carry.y0

        if dense:
            return y1, final_carry.interpolation
        else:
            return y1

    def _integrate_influxsequence(
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
        solver = diffrax.Kvaerno5()
        stepsize_controller = diffrax.PIDController(
            rtol=1e-2,
            atol=1e-2,
            pcoeff=0.1,
            icoeff=0.3,
            jump_ts=jnp.linspace(t0, t1, num=np.int_((t1 - t0) // 4), endpoint=False)
            + 4,
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

        seq_len = np.int_(np.ceil((t1 - t0) // dt))

        return jax.random.bernoulli(key, p=p1, shape=(seq_len,))

    def _sample_influx_events(
        self, t0: float, t1: float, n: int, key: PRNGKeyArray
    ) -> Array:
        return jnp.sort(jax.random.uniform(key, (n,), minval=t0, maxval=t1))

    def integrate(
        self,
        control: controls.AbstractControl,
        state: FibrosisState,
        key: jax.random.KeyArray,
        *,
        max_steps: int = 10000,
        dense: bool = False,
        throw: bool = False,
        ret_influx_events: bool = False,
    ) -> Array:
        t0 = 0.0
        t1 = 1000.0

        influx_events = self._sample_influx_events(t0=t0, t1=t1, n=10, key=key)
        solution = self._integrate_waitingtimes(
            t0=t0,
            t1=t1,
            y0=state.y0,
            control=control,
            influx_events=influx_events,
            saveat=diffrax.SaveAt(ts=jnp.linspace(t0, t1, 1001)),
            max_steps=max_steps,
            throw=throw,
            dense=dense,
        )

        if ret_influx_events:
            return solution, influx_events
        else:
            return solution

        """
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

        return sol.ys
        """
