from typing import Callable

import jax
import jax.numpy as jnp
from jax._src.typing import Array, ArrayLike


@jax.jit
def fibrosis_ode(x: Array, t: ArrayLike, u: Callable[[ArrayLike], Array]) -> Array:
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
    k[12] = 0  # 120 inflammation pulse
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
