from functools import partial

import jax
import jax.numpy as jnp
from jax import lax


@partial(jax.jit, static_argnums=(0,))
def newton_iteration(f, x, *args):
    j_at_x = jax.jacfwd(f)(x, *args)
    f_at_x = f(x, *args)

    delta_x = jnp.linalg.solve(j_at_x, -f_at_x)
    return x + delta_x


@partial(jax.jit, static_argnums=(1,))
def rk4(h, f, y, t, *args):
    k1 = f(y, t, *args)
    k2 = f(y + h * k1 / 2, t + h / 2, *args)
    k3 = f(y + h * k2 / 2, t + h / 2, *args)
    k4 = f(y + h * k3, t + h, *args)

    return y + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4) * h


@partial(jax.jit, static_argnums=(0,))
def odeint_rk4(func, y0, t, *args):
    def scan_func(carry, t):
        y, t_prev = carry
        y_next = rk4(t - t_prev, func, y, t, *args)
        return (y_next, t), y_next

    return lax.scan(scan_func, (y0, t[0]), t)[1]


# @partial(jax.jit, static_argnums=(0,))
# def newton_solve(f, x0, *args, max_iter=100, tol=1e-4):
#    jax.lax.wh


@partial(jax.jit, static_argnums=(1,))
def backward_euler(h, f, y0, t1, *args, num_newton_iters=4):
    # y_1 = y_0 + hf(t_1, y_1)
    # => y_1 - y_0 - hf(t_1, y_1) = 0
    # => solve g(y_1) = 0 for y_1 via Newtons's method

    def g(y1):
        return y1 - y0 - h * f(y1, t1, *args)

    return jax.lax.fori_loop(
        0,
        num_newton_iters,
        lambda i, y1: newton_iteration(g, y1),
        y0,  # + h * f(y0, t1, *args) # Initial guess via Forward Euler
    )


@partial(jax.jit, static_argnums=(0,))
def odeint_backward_euler(func, y0, t, *args):
    def scan_func(carry, t_next):
        y, t = carry
        y_next = backward_euler(t_next - t, func, y, t_next, *args)
        return (y_next, t_next), y_next

    return lax.scan(scan_func, (y0, t[0]), t)[1]


@partial(jax.jit, static_argnums=(1,))
def trapezoidal_rule(h, f, y0, t0, t1, *args, num_newton_iters=4):
    # y_1 = y_0 + 1/2 h (f(t, y) + f(t_1, y_1))
    # => y_1 - y_0 - 1/2 h (f(t, y) + f(t_1, y_1)) = 0
    # => solve g(y_1) = 0 for y_1 via Newtons's method

    def g(y1):
        return y1 - y0 - 0.5 * h * (f(y0, t0, *args) + f(y1, t1, *args))

    return jax.lax.fori_loop(
        0,
        num_newton_iters,
        lambda i, y1: newton_iteration(g, y1),
        y0 + h * f(y0, t0, *args),  # Initial guess via Forward Euler
    )


@partial(jax.jit, static_argnums=(0,))
def odeint_trapezoidal_rule(func, y0, t, *args):
    def scan_func(carry, t_next):
        y, t = carry
        y_next = trapezoidal_rule(t_next - t, func, y, t, t_next, *args)
        return (y_next, t_next), y_next

    return lax.scan(scan_func, (y0, t[0]), t)[1]
