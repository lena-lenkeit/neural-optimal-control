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


def apoptosis_ode(t: ArrayLike, x: Array, args: controls.AbstractControl) -> Array:
    # Constants
    k = [None] * 11

    k[0] = 8.12e-4  # kon,FADD
    k[1] = 0.00567  # koff,FADD
    k[2] = 0.000492  # kon,p55
    k[3] = 0.0114  # kcl,D216
    k[4] = 4.47e-4  # kcl,D374,trans,p55
    k[5] = 0.00344  # kcl,D374,trans,p43
    k[6] = 0.0950  # kp18,inactive
    k[7] = 0.000529  # kcl,BID
    k[8] = 0.00152  # kcl,probe
    k[9] = 8.98  # KD,R
    k[10] = 15.4  # KD,L

    # Control (CD95L)
    control = args
    CD95L = control(t)[0]

    # Active CD95 receptors, steady state solution (in response to CD95L / control)
    CD95act = (
        x[0] ** 3
        * k[10] ** 2
        * CD95L
        / (
            (CD95L + k[10])
            * (
                x[0] ** 2 * k[10] ** 2
                + k[9] * CD95L**2
                + 2 * k[9] * k[10] * CD95L
                + k[9] * k[10] ** 2
            )
        )
    )  # CD95act

    # ODEs
    dx = [None] * 17

    dx[0] = 0  # CD95
    dx[1] = -k[0] * CD95act * x[1] + k[1] * x[6]  # FADD
    dx[2] = -k[2] * x[2] * x[6]  # p55free
    dx[3] = -k[7] * x[3] * (x[9] + x[10])  # Bid
    dx[4] = -k[8] * x[4] * (x[9] + x[10])  # PrNES-mCherry
    dx[5] = -k[8] * x[5] * x[10]  # PrER-mGFP
    dx[6] = (
        k[0] * CD95act * x[1]
        - k[1] * x[6]
        - k[2] * x[2] * x[6]
        + k[3] * x[9]
        + k[4] * x[8] * (x[7] + x[8])
        + (k[5] * x[8] * x[9])
    )  # DISC
    dx[7] = (
        k[2] * x[2] * x[6]
        - k[3] * x[7]
        - k[4] * x[7] * (x[7] + x[8])
        - k[5] * x[7] * x[9]
    )  # DISCp55
    dx[8] = k[3] * x[7] - k[4] * x[8] * (x[7] + x[8]) - k[5] * x[8] * x[9]  # p30
    dx[9] = -k[3] * x[9] + k[4] * x[7] * (x[7] + x[8]) + k[5] * x[7] * x[9]  # p43
    dx[10] = (
        k[3] * x[9] + k[4] * x[8] * (x[7] + x[8]) + k[5] * x[8] * x[9] - k[6] * x[10]
    )  # p18
    dx[11] = k[6] * x[10]  # p18inactive
    dx[12] = k[7] * x[3] * (x[9] + x[10])  # tBid
    dx[13] = k[8] * x[4] * (x[9] + x[10])  # PrNES
    dx[14] = k[8] * x[4] * (x[9] + x[10])  # mCherry
    dx[15] = k[8] * x[5] * x[10]  # PrER
    dx[16] = k[8] * x[5] * x[10]  # mGFP

    return jnp.stack(dx, axis=-1)


class ApoptosisState(environments.EnvironmentState):
    x0: Array


class ApoptosisEnvironment(environments.AbstractEnvironment):
    x0_filepath: str
    x0_split: Tuple[int, int]
    num_cells: int

    def init(self) -> ApoptosisState:
        x0_mat = scipy.io.loadmat(self.x0_filepath)
        x0 = x0_mat["x0_A"][self.x0_split[0] : self.x0_split[1]]
        x0 = jnp.asarray(x0)

        return ApoptosisState(x0=x0)

    def _sample_x0(
        self, state: ApoptosisState, key: jax.random.KeyArray
    ) -> Tuple[Array, Array]:
        idx = jnp.arange(state.x0.shape[0])
        idx = jax.random.choice(key, idx, shape=(self.num_cells,))

        x0 = state.x0[idx]
        x0 = jnp.concatenate(
            (x0[..., 0:4], x0[..., 5:6], x0[..., 4:5], jnp.zeros((self.num_cells, 11))),
            axis=-1,
        )

        return x0, idx

    def integrate(
        self,
        control: controls.AbstractControl,
        state: ApoptosisState,
        key: jax.random.KeyArray,
    ) -> Array:
        def _integrate(y0: Array) -> Array:
            terms = diffrax.ODETerm(apoptosis_ode)
            # solver = diffrax.Kvaerno5()
            solver = diffrax.Dopri5()
            # stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-3)

            sol = diffrax.diffeqsolve(
                terms=terms,
                solver=solver,
                t0=0.0,
                t1=180.0,  # Minutes
                dt0=0.5,
                y0=y0,
                args=control,
                saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, 180.0, 181)),
                # stepsize_controller=stepsize_controller,
                max_steps=100000,
                adjoint=diffrax.RecursiveCheckpointAdjoint(checkpoints=100000),
            )

            return sol.ys

        vintegrate = jax.vmap(_integrate, in_axes=(0,), out_axes=0)
        y0, idx = self._sample_x0(state, key)
        ys = vintegrate(y0)
        # ys = _integrate(y0[0])

        return ys, state.x0[idx, -1] * 1.4897
