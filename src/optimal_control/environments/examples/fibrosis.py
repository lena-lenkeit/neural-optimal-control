from functools import partial
from typing import Callable

import diffrax
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import optimal_control.environments as environments
import optimal_control.solvers as solvers


@partial(jax.jit, static_argnums=(2, 3))
def fibrosis_ode(
    x: Array,
    t: ArrayLike,
    u: Callable[[ArrayLike], Array],
    inflammation_pulse: bool = False,
) -> Array:
    k = {}

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

    u_at_t = u(t)

    # PDGF antibody
    k_pdfg_ab = 1 * 1440  # 1 / (min * molecule)
    pdgf_ab_deg = -k_pdfg_ab * x[3] * u_at_t[0]

    # CSF1 antibody
    k_csf1_ab = 1 * 1440  # 1 / (min * molecule)
    csf1_ab_deg = -k_csf1_ab * x[2] * u_at_t[1]

    # Cytostatic drug
    # k[0] = 0.9 * (1 - u_at_t[1] / (u_at_t[1] + 1.0))

    dx0 = x[0] * (k[0] * x[3] / (k[10] + x[3]) * (1 - x[0] / k[3]) - k[2])  # Fibrobasts
    dx1 = x[1] * (k[1] * x[2] / (k[11] + x[2]) - k[2]) + k[12]  # Mph
    dx2 = (
        csf1_ab_deg + k[6] * x[0] - k[8] * x[1] * x[2] / (k[11] + x[2]) - k[4] * x[2]
    )  # CSF
    dx3 = (
        pdgf_ab_deg
        + k[7] * x[1]
        + k[5] * x[0]
        - k[9] * x[0] * x[3] / (k[10] + x[3])
        - k[4] * x[3]
    )  # PDGF

    return jnp.array([dx0, dx1, dx2, dx3])


class FibrosisEnvironment(environments.AbstractEnvironment):
    def init(self) -> environments.EnvironmentState:
        return {
            "y0": self._integrate(
                0.0,
                300.0,
                jnp.asarray([1.0, 1.0, 0.0, 0.0]),
                lambda _: jnp.asarray([0.0, 0.0]),
                True,
                diffrax.SaveAt(t1=True),
            ).ys
        }

    def _integrate(
        self,
        t0: float,
        t1: float,
        y0: Array,
        u: Callable[[ArrayLike], Array],
        inflammation_pulse: bool,
        saveat: diffrax.SaveAt,
    ) -> diffrax.Solution:
        terms = diffrax.ODETerm(lambda t, y, args: fibrosis_ode(y, t, *args))
        solver = diffrax.Kvaerno5()
        stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=0.0)

        sol = diffrax.diffeqsolve(
            terms,
            solver,
            t0,
            t1,
            0.01,
            y0,
            args=(u, inflammation_pulse),
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=10000,
        )

        return sol

    def integrate(
        self, control: solvers.AbstractControl, state: environments.EnvironmentState
    ) -> Array:
        return self._integrate(
            0.0,
            200.0,
            state["y0"],
            control,
            False,
            diffrax.SaveAt(ts=jnp.linspace(0.0, 200.0, 201)),
        ).ys
