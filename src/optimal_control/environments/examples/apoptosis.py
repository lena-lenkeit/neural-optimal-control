from functools import partial
from typing import Callable, Tuple

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import scipy.io
from jaxtyping import Array, ArrayLike, Scalar

import optimal_control
import optimal_control.controls as controls
import optimal_control.environments as environments


def apoptosis_ode(t: Scalar, y: Array, u: Array, args) -> Array:
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
    CD95L = u[0]

    # Active CD95 receptors, steady state solution (in response to CD95L / control)
    CD95act = (
        y[0] ** 3
        * k[10] ** 2
        * CD95L
        / (
            (CD95L + k[10])
            * (
                y[0] ** 2 * k[10] ** 2
                + k[9] * CD95L**2
                + 2 * k[9] * k[10] * CD95L
                + k[9] * k[10] ** 2
            )
        )
    )  # CD95act

    # ODEs
    dx = [None] * 17

    dx[0] = 0  # CD95
    dx[1] = -k[0] * CD95act * y[1] + k[1] * y[6]  # FADD
    dx[2] = -k[2] * y[2] * y[6]  # p55free
    dx[3] = -k[7] * y[3] * (y[9] + y[10])  # Bid
    dx[4] = -k[8] * y[4] * (y[9] + y[10])  # PrNES-mCherry
    dx[5] = -k[8] * y[5] * y[10]  # PrER-mGFP
    dx[6] = (
        k[0] * CD95act * y[1]
        - k[1] * y[6]
        - k[2] * y[2] * y[6]
        + k[3] * y[9]
        + k[4] * y[8] * (y[7] + y[8])
        + (k[5] * y[8] * y[9])
    )  # DISC
    dx[7] = (
        k[2] * y[2] * y[6]
        - k[3] * y[7]
        - k[4] * y[7] * (y[7] + y[8])
        - k[5] * y[7] * y[9]
    )  # DISCp55
    dx[8] = k[3] * y[7] - k[4] * y[8] * (y[7] + y[8]) - k[5] * y[8] * y[9]  # p30
    dx[9] = -k[3] * y[9] + k[4] * y[7] * (y[7] + y[8]) + k[5] * y[7] * y[9]  # p43
    dx[10] = (
        k[3] * y[9] + k[4] * y[8] * (y[7] + y[8]) + k[5] * y[8] * y[9] - k[6] * y[10]
    )  # p18
    dx[11] = k[6] * y[10]  # p18inactive
    dx[12] = k[7] * y[3] * (y[9] + y[10])  # tBid
    dx[13] = k[8] * y[4] * (y[9] + y[10])  # PrNES
    dx[14] = k[8] * y[4] * (y[9] + y[10])  # mCherry
    dx[15] = k[8] * y[5] * y[10]  # PrER
    dx[16] = k[8] * y[5] * y[10]  # mGFP

    return jnp.stack(dx, axis=-1)


class ApoptosisState(environments.EnvironmentState):
    x0: Array


class ApoptosisEnvironment(environments.AbstractEnvironment):
    x0_filepath: str
    x0_split: Tuple[int, int]
    num_cells: int
    full_split: bool = False

    def init(self) -> ApoptosisState:
        x0_mat = scipy.io.loadmat(self.x0_filepath)
        x0 = x0_mat["x0_A"][self.x0_split[0] : self.x0_split[1]]
        x0 = jnp.asarray(x0)

        return ApoptosisState(x0=x0)

    def _sample_x0(
        self, state: ApoptosisState, key: jax.random.KeyArray
    ) -> Tuple[Array, Array]:
        if self.full_split:
            idx = jnp.arange(state.x0.shape[0])
        else:
            idx = jax.random.randint(
                key, (self.num_cells,), minval=0, maxval=state.x0.shape[0]
            )

        x0 = state.x0[idx]
        x0 = jnp.concatenate(
            (x0[..., [0, 1, 2, 3, 5, 4]], jnp.zeros((len(idx), 11))),
            axis=-1,
        )

        return x0, idx

    def integrate(
        self,
        control: controls.AbstractControl,
        state: ApoptosisState,
        key: jax.random.KeyArray,
        *,
        t1: Scalar = 180.0,
        max_steps: int = 181,
        saveat: diffrax.SaveAt = diffrax.SaveAt(ts=jnp.linspace(0.0, 180.0, 181)),
        vmap: str = "outer"
    ) -> Tuple[diffrax.Solution, Array]:
        if vmap == "outer":

            def _integrate(y0: Array) -> diffrax.Solution:
                ode = apoptosis_ode
                ode = optimal_control.with_control(ode, time=True)

                terms = diffrax.ODETerm(ode)
                solver = diffrax.Dopri5()

                sol = diffrax.diffeqsolve(
                    terms=terms,
                    solver=solver,
                    t0=0.0,
                    t1=t1,  # Minutes
                    dt0=1.0,
                    y0=y0,
                    args=(control, None),
                    saveat=saveat,
                    max_steps=max_steps,
                    adjoint=diffrax.RecursiveCheckpointAdjoint(checkpoints=max_steps),
                )

                return sol

            vintegrate = jax.vmap(_integrate, in_axes=(0,), out_axes=0)
            y0, idx = self._sample_x0(state, key)
            solution = vintegrate(y0)

            return solution, state.x0[idx, -1] * 1.4897
        elif vmap == "inner":

            def _integrate(y0: Array) -> diffrax.Solution:
                ode = eqx.filter_vmap(
                    apoptosis_ode, in_axes=(None, 0, None, None), out_axes=0
                )
                ode = optimal_control.with_control(ode, time=True)

                terms = diffrax.ODETerm(ode)
                solver = diffrax.Dopri5()

                sol = diffrax.diffeqsolve(
                    terms=terms,
                    solver=solver,
                    t0=0.0,
                    t1=t1,  # Minutes
                    dt0=1.0,
                    y0=y0,
                    args=(control, None),
                    saveat=saveat,
                    max_steps=max_steps,
                    adjoint=diffrax.RecursiveCheckpointAdjoint(checkpoints=max_steps),
                )

                return sol

            y0, idx = self._sample_x0(state, key)
            solution = _integrate(y0)

            return solution, state.x0[idx, -1] * 1.4897
        else:
            raise ValueError(vmap)
