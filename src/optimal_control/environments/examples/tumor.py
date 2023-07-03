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


def tumor_ode(t: Scalar, y: PyTree, u: PyTree, args: PyTree) -> PyTree:
    tumor_growth_rate = 1.2
    tumor_death_rate = 1.0
    healthy_growth_rate = 1.1
    healthy_death_rate = 1.0
    compartment_capacity = 1000.0
    nutrient_influx = 100.0

    k_nutrients_tumor = 2.0
    k_nutrients_healthy = 1.0

    tumor_cells, healthy_cells, nutrients = jnp.split(y, 3, axis=-1)

    total_cells = tumor_cells + healthy_cells
    compartment_capacity_factor = 1 - (total_cells) / compartment_capacity

    tumor_nutrient_factor = nutrients / (k_nutrients_tumor + nutrients)
    healthy_nutrient_factor = nutrients / (k_nutrients_healthy + nutrients)

    d_tumor_cells_dt = (
        tumor_growth_rate
        * compartment_capacity_factor
        * tumor_cells
        * tumor_nutrient_factor
        - tumor_cells * tumor_death_rate
    )
    d_healthy_cells_dt = (
        healthy_growth_rate
        * compartment_capacity_factor
        * healthy_cells
        * healthy_nutrient_factor
        - healthy_cells * healthy_death_rate
    )
    d_nutrients_dt = (
        nutrient_influx
        - tumor_cells * tumor_nutrient_factor
        - healthy_cells * healthy_nutrient_factor
    )

    return jnp.stack((d_tumor_cells_dt, d_healthy_cells_dt, d_nutrients_dt), axis=-1)
