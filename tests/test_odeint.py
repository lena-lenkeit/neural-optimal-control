import unittest

import jax
import jax.numpy as jnp

from optimal_control import odeint


class TestNumericalIntegrators(unittest.TestCase):
    def setUp(self):
        @jax.jit
        def f(x, t):
            return -x

        self.f = f
        self.t = jnp.linspace(0.0, 10.0, 100)
        self.x0 = jnp.array([1.0])
        self.x1 = jnp.exp(-self.t[-1])

    def test_rk4(self):
        xt = odeint.odeint_rk4(self.f, self.x0, self.t)
        x1 = xt[-1]

        self.assertTrue(jnp.abs(x1 - self.x1) < 0.01)

    def test_backward_euler(self):
        xt = odeint.odeint_backward_euler(self.f, self.x0, self.t)
        x1 = xt[-1]

        self.assertTrue(jnp.abs(x1 - self.x1) < 0.01)

    def test_trapezoidal_rule(self):
        xt = odeint.odeint_trapezoidal_rule(self.f, self.x0, self.t)
        x1 = xt[-1]

        self.assertTrue(jnp.abs(x1 - self.x1) < 0.01)


class TestRootFinders(unittest.TestCase):
    def setUp(self):
        @jax.jit
        def f(x):
            return x**2

        self.f = f
        self.x0 = jnp.array([2.0])
        self.x1 = jnp.array([0.0])

    def test_newton_iteration(self):
        x1 = jax.lax.fori_loop(
            0, 10, lambda i, xi: odeint.newton_iteration(self.f, xi), self.x0
        )

        self.assertTrue(jnp.abs(x1 - self.x1) < 0.01)


if __name__ == "__main__":
    unittest.main()
