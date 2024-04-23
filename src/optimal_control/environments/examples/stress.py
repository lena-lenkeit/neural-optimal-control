from functools import partial
from typing import Callable, Tuple

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import scipy.io
from jaxtyping import Array, ArrayLike, PRNGKeyArray, PyTree, Scalar

import optimal_control
import optimal_control.controls as controls
import optimal_control.environments as environments


def det_ThaKin_ld_C1_G0_1K_wP_kd_wp_67BF35A2(
    t: Scalar, y: Array, u: Array, args: Array
) -> Array:
    k, tha_mult = args

    u = [u[0] * tha_mult]

    # ODE
    dy = [None] * 10

    a = [y[1] ** k[4] / (k[5] ** k[4] + y[1] ** k[4])]  # Tr_inh

    dy[0] = (
        -k[0] * y[0]
        - (k[1] * u[0] / (k[2] + u[0]) * y[0] / (k[3] + y[0]))
        + k[10] * y[1] * y[3]
        + k[11] * y[1]
    )  # eIF2a
    dy[1] = (
        k[0] * y[0]
        + (k[1] * u[0] / (k[2] + u[0]) * y[0] / (k[3] + y[0]))
        - k[10] * y[1] * y[3]
        - k[11] * y[1]
    )  # p_eIF2a
    dy[2] = k[6] * y[9] - (k[7] * y[2])  # m_GADD34
    dy[3] = k[8] * y[2] - (k[9] * y[3])  # GADD34
    dy[4] = -k[12] * y[4] * a[0] + (k[13] * y[9])  # Pr_tot
    dy[5] = k[12] * y[4] * a[0] - (k[12] * y[5])  # Pr_delay_1
    dy[6] = k[12] * y[5] - (k[12] * y[6])  # Pr_delay_2
    dy[7] = k[12] * y[6] - (k[12] * y[7])  # Pr_delay_3
    dy[8] = k[12] * y[7] - (k[12] * y[8])  # Pr_delay_4
    dy[9] = k[12] * y[8] - (k[13] * y[9])  # Pr_delay_5

    return jnp.stack(dy, axis=-1)


def f_sg(p_eif2a: Array, h_sg: Scalar, k_sg: Scalar) -> Array:
    return p_eif2a**h_sg / (k_sg**h_sg + p_eif2a**h_sg)


class StressState(environments.EnvironmentState):
    x0: Array
    s0: Array
    k: Array


class StressEnvironment(environments.AbstractEnvironment):
    couples_filepath: str
    couple_idx: int = -1
    use_updated_params: bool = False

    def init(self) -> StressState:
        # Load couple
        matfile = scipy.io.loadmat(self.couples_filepath)
        couples = matfile["couples"][0]
        couple = couples[self.couple_idx]

        # Import data from couple
        x0_idx = 11
        k_idx = 9

        x0 = couple[x0_idx].flatten()
        k = couple[k_idx].flatten()

        # Replace with updated parameters
        if self.use_updated_params:
            k[[1, 2, 3]] = [0.1303e03, 0.3310e03, 5.2018e03]

        # To jax
        x0 = jnp.asarray(x0)
        k = jnp.asarray(k)

        # h_sg_idx = 4
        # k_sg_tha_idx = 5

        # h_sg = k[h_sg_idx]
        # k_sg_tha = k[k_sg_tha_idx]

        # p_eif2a_idx = 1

        # Get initial state
        control = controls.LambdaControl(lambda _: jnp.zeros((1,)))
        s0 = self._integrate(
            control=control,
            parameters=k,
            t0=0,
            t1=10 * 24 * 60,
            y0=x0,
            saveat=diffrax.SaveAt(t1=True),
        ).ys[-1]

        return StressState(x0, s0, k)

    def _integrate(
        self,
        control: controls.AbstractControl,
        parameters: Array,
        t0: Scalar,
        t1: Scalar,
        y0: Array,
        saveat: diffrax.SaveAt,
        stepsize_controller: diffrax.AbstractStepSizeController = diffrax.PIDController(
            atol=1e-5,
            rtol=1e-5,
            pcoeff=1.0,
            icoeff=1.0,
            dtmax=30,
        ),
        tha_mult: Scalar = 1.0,
    ) -> diffrax.Solution:
        ode = det_ThaKin_ld_C1_G0_1K_wP_kd_wp_67BF35A2
        ode = optimal_control.with_control(ode, time=True)
        terms = diffrax.ODETerm(ode)

        solver = diffrax.Kvaerno5()

        sol = diffrax.diffeqsolve(
            terms=terms,
            solver=solver,
            t0=t0,
            t1=t1,
            dt0=0.1,
            y0=y0,
            args=(control, (parameters, tha_mult)),
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=10000,
        )

        return sol

    def integrate(
        self,
        control: controls.AbstractControl,
        state: StressState,
        key: PRNGKeyArray,
        *,
        t1: Scalar = 20 * 60,
        stepsize_controller: diffrax.AbstractStepSizeController = diffrax.PIDController(
            atol=1e-5,
            rtol=1e-5,
            pcoeff=1.0,
            icoeff=1.0,
            dtmax=30,
        ),
        tha_lognormal_mean: Scalar = 0.0,  # 0.40546510810816438197801311546435,
        tha_lognormal_std: Scalar = 0.0,
        k_lognormal_std: ArrayLike = 0.0,
        s0_lognormal_std: ArrayLike = 0.0,
        zero_control: bool = True,
    ) -> Tuple[Array, Array]:
        key, subkey = jax.random.split(key)
        tha_mult = 1.0
        tha_mult *= jnp.exp(
            tha_lognormal_mean + jax.random.normal(subkey) * tha_lognormal_std
        )

        key, subkey = jax.random.split(key)
        k = state.k
        k = k * jnp.exp(jax.random.normal(subkey, k.shape) * k_lognormal_std)

        key, subkey = jax.random.split(key)
        s0 = state.s0
        s0 = s0 * jnp.exp(jax.random.normal(subkey, s0.shape) * s0_lognormal_std)

        ts = jnp.linspace(0.0, t1, int(t1) + 1)
        ts1 = ts[: ts.shape[0] // 2 + 1]
        ts2 = ts[ts.shape[0] // 2 + 1 :]

        sol1 = self._integrate(
            control=control,
            parameters=k,
            t0=ts1[0],
            t1=ts1[-1],
            y0=s0,
            saveat=diffrax.SaveAt(ts=ts1),
            stepsize_controller=stepsize_controller,
            tha_mult=tha_mult,
        )

        sol2 = self._integrate(
            control=(
                controls.LambdaControl(lambda _: jnp.zeros(1))
                if zero_control
                else control
            ),
            parameters=k,
            t0=ts1[-1],
            t1=ts2[-1],
            y0=sol1.ys[-1],
            saveat=diffrax.SaveAt(ts=ts2),
            stepsize_controller=stepsize_controller,
            tha_mult=tha_mult,
        )

        ys = jnp.concatenate((sol1.ys, sol2.ys), axis=0)
        sg = f_sg(ys[:, 1], k[4], k[5])

        return ys, sg
