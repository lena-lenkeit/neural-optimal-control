from functools import partial
from typing import Callable, List, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree

from optimal_control.utils import exists


class AbstractControl(eqx.Module):
    r""""""


class LambdaControl(AbstractControl):
    control_fun: Union[
        Callable[[ArrayLike, PyTree], Array], Callable[[ArrayLike], Array]
    ]
    data: Optional[PyTree] = None

    def __call__(self, t: ArrayLike) -> Array:
        if exists(self.data):
            return self.control_fun(t, self.data)
        else:
            return self.control_fun(t)


class InterpolationControl(AbstractControl):
    control: Array
    channels: int
    steps: int
    t_start: float
    t_end: float
    method: str

    def __init__(
        self,
        channels: int,
        steps: int,
        t_start: float,
        t_end: float,
        method: str = "linear",
        control: Optional[Array] = None,
    ):
        self.channels = channels
        self.steps = steps
        self.t_start = t_start
        self.t_end = t_end
        self.method = method

        if not exists(control):
            self.control = jnp.zeros((self.steps, self.channels))
        else:
            self.control = control

    @staticmethod
    def interpolate_linear(x: ArrayLike, xp: ArrayLike, fp: ArrayLike) -> Array:
        vintp = jax.vmap(
            partial(jnp.interp, left=0.0, right=0.0),
            in_axes=(None, None, -1),
            out_axes=-1,
        )

        return vintp(x, xp, fp)

    @staticmethod
    def interpolate_step(x: ArrayLike, xp: ArrayLike, fp: ArrayLike) -> Array:
        def interp(x: ArrayLike, xp: ArrayLike, fp: ArrayLike) -> Array:
            idx = jnp.searchsorted(xp, x)
            y = jnp.where((x <= xp[0]) | (x > xp[-1]), 0.0, fp[idx - 1])

            return y

        vintp = jax.vmap(interp, in_axes=(None, None, -1), out_axes=-1)
        return vintp(x, xp, fp)

    @staticmethod
    def interpolate(x: ArrayLike, xp: ArrayLike, fp: ArrayLike, method: str) -> Array:
        if method == "linear":
            return InterpolationControl.interpolate_linear(x, xp, fp)
        elif method == "step":
            return InterpolationControl.interpolate_step(x, xp, fp)

    @staticmethod
    @jax.jit
    @partial(jax.vmap, in_axes=(None, -1, None, None), out_axes=-1)
    # Weirdly, this is much slower in the Fibrosis environment
    #   Maybe because it's a step function?
    def fast_interpolate_step(t: ArrayLike, c: Array, t0: float, t1: float) -> Array:
        # Find indicies into array
        i = (t - t0) / (t1 - t0)
        i = jnp.floor(i * c.shape[0]).astype(jnp.int32)

        # Replace left OOB indices
        #   We want all OOB indices to return zeros in the gather
        #   OOB indices on the right count as OOB, hence this works trivially
        #   OOB indices on the left act as reverse indices, hence we force them OOB
        i = jnp.where(i < 0, c.shape[0], i)

        # Gather array, replacing OOB indices with 0
        x = c.at[i].get(mode="fill", fill_value=0.0)
        return x

    @staticmethod
    @jax.jit
    @partial(jax.vmap, in_axes=(None, -1, None, None), out_axes=-1)
    def fast_interpolate_linear(t: ArrayLike, c: Array, t0: float, t1: float) -> Array:
        # Find continuous indices
        ci = (t - t0) / (t1 - t0)
        ci = ci * c.shape[0]

        # Extract left / right indices and interpolant
        li = jnp.floor(ci)
        ri = li + 1
        p = ci - li

        li = li.astype(jnp.int32)
        ri = ri.astype(jnp.int32)

        # Get values at indices
        lx = c.at[li].get()
        rx = c.at[ri].get()

        # Interpolate
        ix = p * (rx - lx) + lx

        # Clip
        ix = jnp.where((li < 0) | (ri >= c.shape[0]), 0, ix)

        return ix

    def __call__(self, t: ArrayLike) -> Array:
        """
        if self.method == "step":
            return InterpolationControl.fast_interpolate_step(
                t, self.control, self.t_start, self.t_end
            )
        elif self.method == "linear":
            return InterpolationControl.fast_interpolate_linear(
                t, self.control, self.t_start, self.t_end
            )
        """

        # """
        t = (t - self.t_start) / (self.t_end - self.t_start)
        return InterpolationControl.interpolate(
            t, jnp.linspace(0.0, 1.0, self.steps), self.control, self.method
        )
        # """


class ImplicitControl(AbstractControl):
    control: eqx.Module
    t_start: float
    t_end: float

    def __call__(self, t: ArrayLike) -> Array:
        # Rescale t to [-1, 1]
        t = (t - self.t_start) / (self.t_end - self.t_start)
        t = t * 2 - 1

        # Evaluate & clip to zero past borders
        c = self.control(t)
        c = jnp.where((t < -1) | (t > 1), 0.0, c)

        return c


class SineLayer(eqx.Module):
    weight: Array
    bias: Array
    omega: float
    is_linear: bool

    def __init__(
        self,
        in_features: int,
        out_features: int,
        key: jax.random.KeyArray,
        is_first: bool = False,
        is_linear: bool = False,
        omega: float = 30.0,
    ):
        self.omega = omega
        self.is_linear = is_linear

        # Init linear layer
        key1, key2 = jax.random.split(key, 2)

        if is_first:
            init_val = 1 / in_features
        else:
            init_val = (6 / in_features) ** 0.5 / omega

        self.weight = jax.random.uniform(
            key1, (out_features, in_features), minval=-init_val, maxval=init_val
        )

        # self.weight_g = jnp.linalg.norm(weight, ord=2, axis=1, keepdims=True)
        # self.weight_v = weight

        if is_linear:
            self.bias = jnp.zeros(out_features)
        else:
            # self.bias = jax.random.uniform(
            #    key2, (out_features,), minval=-init_val, maxval=init_val
            # )

            init_val = jnp.pi / jnp.linalg.norm(
                self.weight, ord=2, axis=1, keepdims=False
            )

            self.bias = jax.random.uniform(
                key2, (out_features,), minval=-init_val, maxval=init_val
            )

    def __call__(self, x: Array) -> Array:
        # weight = (
        #    self.weight_g
        #    / jnp.linalg.norm(self.weight_v, ord=2, axis=1, keepdims=True)
        #    * self.weight_v
        # )

        x = self.weight @ x + self.bias  # Linear layer
        if not self.is_linear:
            x = jnp.sin(self.omega * x)  # Sine activation

        return x


class Siren(eqx.Module):
    layers: List[SineLayer]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        hidden_layers: int,
        key: jax.random.KeyArray,
        first_omega: float = 30.0,
        hidden_omega: float = 30.0,
    ):
        self.layers = []

        for i in range(hidden_layers + 1):
            layer_in_features = hidden_features
            layer_out_features = hidden_features
            layer_omega = hidden_omega

            is_first = i == 0
            is_last = i == hidden_layers

            if is_first:
                layer_in_features = in_features
                layer_omega = first_omega
            if is_last:
                layer_out_features = out_features

            key, layer_key = jax.random.split(key)

            self.layers.append(
                SineLayer(
                    in_features=layer_in_features,
                    out_features=layer_out_features,
                    key=layer_key,
                    is_first=is_first,
                    is_linear=is_last,
                    omega=layer_omega,
                )
            )

    def __call__(self, x: Array) -> Array:
        for layer in self.layers:
            x = layer(x)

        return x
