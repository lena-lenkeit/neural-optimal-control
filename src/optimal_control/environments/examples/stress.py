from functools import partial
from typing import Callable, Tuple

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import scipy.io
from jaxtyping import Array, ArrayLike

import optimal_control.controls as controls
import optimal_control.environments as environments


def det_ThaKin_ld_C1_G0_1K_wP_kd_wp_67BF35A2(t, x, args):
    k, u = args

    # ODE
    dx = [None] * 10

    u = u(t)
    a = [x[1] ** k[4] / (k[5] ** k[4] + x[1] ** k[4])]  # Tr_inh

    dx[0] = (
        -k[0] * x[0]
        - (k[1] * u[0] / (k[2] + u[0]) * x[0] / (k[3] + x[0]))
        + k[10] * x[1] * x[3]
        + k[11] * x[1]
    )  # eIF2a
    dx[1] = (
        k[0] * x[0]
        + (k[1] * u[0] / (k[2] + u[0]) * x[0] / (k[3] + x[0]))
        - k[10] * x[1] * x[3]
        - k[11] * x[1]
    )  # p_eIF2a
    dx[2] = k[6] * x[9] - (k[7] * x[2])  # m_GADD34
    dx[3] = k[8] * x[2] - (k[9] * x[3])  # GADD34
    dx[4] = -k[12] * x[4] * a[0] + (k[13] * x[9])  # Pr_tot
    dx[5] = k[12] * x[4] * a[0] - (k[12] * x[5])  # Pr_delay_1
    dx[6] = k[12] * x[5] - (k[12] * x[6])  # Pr_delay_2
    dx[7] = k[12] * x[6] - (k[12] * x[7])  # Pr_delay_3
    dx[8] = k[12] * x[7] - (k[12] * x[8])  # Pr_delay_4
    dx[9] = k[12] * x[8] - (k[13] * x[9])  # Pr_delay_5

    return jnp.stack(dx, axis=-1)


def f_sg(p_eif2a, h_sg, k_sg):
    return p_eif2a**h_sg / (k_sg**h_sg + p_eif2a**h_sg)


class StressState(environments.EnvironmentState):
    x0: Array
    s0: Array
    k: Array


class StressEnvironment(environments.AbstractEnvironment):
    def init(
        self,
        couples_filepath: str,
        couple_idx: int = -1,
    ) -> StressState:
        # Load couple
        matfile = scipy.io.loadmat(couples_filepath)
        couples = matfile["couples"][0]
        couple = couples[couple_idx]

        # Import data from couple
        x0_idx = 11
        k_idx = 9

        x0 = couple[x0_idx].flatten()
        k = couple[k_idx].flatten()

        # h_sg_idx = 4
        # k_sg_tha_idx = 5

        # h_sg = k[h_sg_idx]
        # k_sg_tha = k[k_sg_tha_idx]

        # p_eif2a_idx = 1

        # Get initial state
        control = controls.LambdaControl(lambda t: jnp.zeros((1,)))
        s0 = self._integrate(
            control=control,
            parameters=k,
            t1=10 * 24 * 60,
            y0=x0,
            saveat=diffrax.SaveAt(t1=True),
        ).ys[-1]

        return StressState(x0, s0, k)

    def _integrate(
        self,
        control: controls.AbstractControl,
        parameters: Array,
        t1: ArrayLike,
        y0: Array,
        saveat: diffrax.SaveAt,
    ) -> diffrax.Solution:
        terms = diffrax.ODETerm(det_ThaKin_ld_C1_G0_1K_wP_kd_wp_67BF35A2)
        solver = diffrax.Kvaerno5()
        stepsize_controller = diffrax.PIDController(
            atol=1e-5, rtol=1e-5, pcoeff=0.3, icoeff=0.3
        )

        sol = diffrax.diffeqsolve(
            terms=terms,
            solver=solver,
            t0=0.0,
            t1=t1,
            dt0=0.1,
            y0=y0,
            args=(parameters, control),
            saveat=saveat,
            stepsize_controller=stepsize_controller,
        )

        return sol

    def integrate(
        self,
        control: controls.AbstractControl,
        state: StressState,
        key: jax.random.KeyArray,
    ) -> Tuple[Array, Array]:
        sol = self._integrate(
            control=control,
            parameters=state.k,
            t1=10 * 60,
            y0=state.s0,
            saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, 10 * 60, 10 * 60 + 1)),
        )

        ys = sol.ys
        sg = f_sg(ys[:, 1], state.k[4], state.k[5])

        return ys, sg
