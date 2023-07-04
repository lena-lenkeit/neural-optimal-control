import abc
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PRNGKeyArray, PyTree, Scalar

from optimal_control.controls import AbstractControl
from optimal_control.utils import exists


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
        t = jnp.atleast_1d(t)[0]

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
        # """
        if self.method == "step":
            return InterpolationControl.fast_interpolate_step(
                t, self.control, self.t_start, self.t_end
            )
        elif self.method == "linear":
            return InterpolationControl.fast_interpolate_linear(
                t, self.control, self.t_start, self.t_end
            )
        # """

        """
        t = (t - self.t_start) / (self.t_end - self.t_start)
        return InterpolationControl.interpolate(
            t, jnp.linspace(0.0, 1.0, self.steps), self.control, self.method
        )
        """


class ImplicitControl(AbstractControl):
    control: eqx.Module
    t_start: float
    t_end: float

    def __call__(self, t: ArrayLike) -> Array:
        t = jnp.atleast_1d(t)

        # Rescale t to [-1, 1]
        t = (t - self.t_start) / (self.t_end - self.t_start)
        t = t * 2 - 1

        # Evaluate & clip to zero past borders
        c = self.control(t)
        c = jnp.where((t < -1) | (t > 1), 0.0, c)

        return c


class SineLayer(eqx.Module):
    weight: Array
    # weight_v: Array
    # weight_g: Array
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


class ActiveControl(AbstractControl):
    control: eqx.Module
    t_start: float
    t_end: float


class RNN(eqx.Module):
    in_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    cells: eqx.Module
    cell_type: str
    initial_state: PyTree

    def __init__(
        self,
        in_width: int,
        out_width: int,
        rnn_width: int,
        rnn_layers: int,
        cell_type: str,
        key: PRNGKeyArray,
    ):
        key, key1, key2 = jax.random.split(key, num=3)
        self.in_proj = eqx.nn.Linear(in_width, rnn_width, key=key1)
        self.out_proj = eqx.nn.Linear(rnn_width, out_width, use_bias=False, key=key2)

        self.cell_type = cell_type
        cell_cls = {"gru": eqx.nn.GRUCell, "lstm": eqx.nn.LSTMCell}[cell_type]

        keys = jax.random.split(key, num=rnn_layers)
        make_cells = lambda k: cell_cls(
            input_size=rnn_width, hidden_size=rnn_width, key=k
        )
        self.cells = eqx.filter_vmap(make_cells)(keys)

        if cell_type == "lstm":
            self.initial_state = (
                jnp.zeros((rnn_layers, rnn_width)),
                jnp.zeros((rnn_layers, rnn_width)),
            )
        elif cell_type == "gru":
            self.initial_state = jnp.zeros((rnn_layers, rnn_width))

    def __call__(self, inputs: Array, states: PyTree) -> Tuple[Array, PyTree]:
        x = self.in_proj(inputs)

        # Scan over stack of RNN cells
        cells_jaxtypes, cell_pytypes = eqx.partition(self.cells, eqx.is_array)

        def f(carry: Array, x: Tuple[eqx.Module, PyTree]) -> Tuple[Array, PyTree]:
            input = carry
            cell_jaxtypes, state = x

            cell = eqx.combine(cell_jaxtypes, cell_pytypes)
            next_state = cell(input, state)

            if self.cell_type == "lstm":
                output, _ = next_state  # LSTM-like
            elif self.cell_type == "gru":
                output = next_state  # GRU-like

            return output, next_state

        x, next_states = jax.lax.scan(f, init=x, xs=(cells_jaxtypes, states))

        x = self.out_proj(x)
        return x, next_states


class ModularControl(eqx.Module):
    main: eqx.Module
    encoder: Optional[eqx.Module] = None
    decoder: Optional[eqx.Module] = None
    num_controls: int
    num_latents: Optional[int] = None
    num_states: Optional[int] = None
    mode: str

    def __init__(
        self,
        hidden_width: int,
        hidden_layers: int,
        num_controls: int,
        num_latents: Optional[int],
        num_states: Optional[int],
        rnn_cell_type: Optional[str],
        mode: str = "cde-rnn",
        *,
        key: PRNGKeyArray
    ):
        self.num_controls = num_controls
        self.num_latents = num_latents
        self.num_states = num_states
        self.mode = mode

        if mode == "cde-rnn":
            keys = jax.random.split(key, num=3)
            self.main = eqx.nn.MLP(
                in_size=num_latents,
                out_size=(num_latents * (1 + num_states)),
                width_size=hidden_width,
                depth=hidden_layers,
                activation=jax.nn.silu,
                final_activation=jax.nn.tanh,
                use_final_bias=False,
                key=keys[0],
            )
            self.encoder = eqx.nn.MLP(
                in_size=(1 + num_states),
                out_size=num_latents,
                width_size=hidden_width,
                depth=hidden_layers,
                activation=jax.nn.silu,
                use_final_bias=False,
                key=keys[1],
            )
            self.decoder = eqx.nn.MLP(
                in_size=num_latents,
                out_size=num_controls,
                width_size=hidden_width,
                depth=hidden_layers,
                activation=jax.nn.silu,
                final_activation=jax.nn.tanh,
                use_final_bias=False,
                key=keys[2],
            )
        if mode == "step-rnn":
            self.main = RNN(
                in_width=num_states,
                out_width=num_controls,
                rnn_width=hidden_width,
                rnn_layers=hidden_layers,
                cell_type=rnn_cell_type,
                key=key,
            )

        elif mode == "derivative":
            keys = jax.random.split(key, num=2)
            self.main = eqx.nn.MLP(
                in_size=num_states,
                out_size=num_controls,
                width_size=hidden_width,
                depth=hidden_layers,
                activation=jax.nn.silu,
                use_final_bias=False,
                key=keys[0],
            )
            self.encoder = eqx.nn.MLP(
                in_size=(1 + num_states),
                out_size=num_controls,
                width_size=hidden_width,
                depth=hidden_layers,
                activation=jax.nn.silu,
                use_final_bias=False,
                key=keys[1],
            )

    def __call__(
        self, inputs: Array, states: Optional[PyTree] = None
    ) -> Union[Array, Tuple[Array, PyTree]]:
        if self.mode == "cde-rnn":
            dzdX: Array = self.main(inputs)
            dzdX = dzdX.reshape(self.num_latents, 1 + self.num_states)

            return dzdX
        elif self.mode == "step-rnn":
            return self.main(inputs, states)
        else:
            return self.main(inputs)

    def encode_controls(self, X0: Array) -> Array:
        return self.encoder(X0)

    def encode_latents(self, z0: Array) -> Array:
        return self.encoder(z0)

    def decode_latents(self, z: Array) -> Array:
        return self.decoder(z)
