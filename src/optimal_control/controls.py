import abc
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PRNGKeyArray, PyTree, Scalar

from optimal_control.utils import exists


class AbstractControl(eqx.Module):
    ...
