import functools
import math
from typing import Any, Callable, Optional, Tuple

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree, Scalar


def with_control(
    f: Callable[[Scalar, PyTree, PyTree, PyTree], PyTree],
    time: bool = False,
    state: bool = False,
) -> Callable[[Scalar, PyTree, PyTree], PyTree]:
    @functools.wraps(f)
    def wrapper(t: Scalar, y: PyTree, args: PyTree) -> PyTree:
        control, f_args = args

        if time:
            u = control(t=t)
        if state:
            u = control(y=y)

        dy = f(t, y, u, f_args)
        return dy

    def modify_initial_state(control: PyTree, t0: Scalar, y0: Array) -> Array:
        if hasattr(f, "_modify_initial_state"):
            return f._modify_initial_state(control, t0, y0)
        else:
            return y0

    wrapper_fn = wrapper
    wrapper_fn._modify_initial_state = modify_initial_state

    return wrapper_fn


def with_derivative_control(
    f: Callable[[Scalar, PyTree, PyTree, PyTree], PyTree], num_controls: int
) -> Callable[[Scalar, PyTree, PyTree], PyTree]:
    @functools.wraps(f)
    def wrapper(t: Scalar, y: PyTree, args: PyTree) -> PyTree:
        control, f_args = args
        c_u, f_y = y[..., :num_controls], y[..., num_controls:]

        f_dy = f(t, f_y, c_u, f_args)
        c_du = control(y)

        dy = jnp.concatenate((c_du, f_dy), axis=-1)
        return dy

    def modify_initial_state(control: PyTree, t0: Scalar, y0: Array) -> Array:
        X0 = jnp.concatenate((t0, y0), axis=-1)
        c0 = control.encode_controls(X0)

        y0 = jnp.concatenate((c0, y0), axis=-1)
        return y0

    wrapper_fn = wrapper
    wrapper_fn._modify_initial_state = modify_initial_state

    return wrapper_fn


# This needs stepping support, since ODE RNNs are discontinuous over state changes
# def with_ode_rnn_control(f: Callable[[Scalar, PyTree, PyTree, PyTree], PyTree], num_controls: int, num_memory: int):


def with_cde_rnn_control(
    f: Callable[[Scalar, PyTree, PyTree, PyTree], PyTree], num_latents: int
):
    @functools.wraps(f)
    def wrapper(t: Scalar, y: PyTree, args: PyTree) -> PyTree:
        control, f_args = args
        f_y, c_z = y[..., :-num_latents], y[..., -num_latents:]

        c_u = control.decode_latents(c_z)
        f_dy = f(t, f_y, c_u, f_args)

        dt = jnp.ones(1)
        dX = jnp.concatenate((dt, f_dy), axis=-1)

        # c_dzdX = control(c_z)
        # c_dz = c_dzdX @ dX

        c_dz = control(c_z, dX, f_y)

        dy = jnp.concatenate((f_dy, c_dz), axis=-1)
        return dy

    def modify_initial_state(control: PyTree, t0: Scalar, y0: Array) -> Array:
        if hasattr(f, "_modify_initial_state"):
            y0 = f._modify_initial_state(control, t0, y0)

        X0 = jnp.concatenate((jnp.atleast_1d(t0), y0), axis=-1)
        z0 = control.encode_latents(X0)

        y0 = jnp.concatenate((y0, z0), axis=-1)
        return y0

    wrapper_fn = wrapper
    wrapper_fn._modify_initial_state = modify_initial_state

    return wrapper_fn


def with_extra_term(
    f: Callable[[Scalar, PyTree, PyTree, PyTree], PyTree],
    g: Callable[[Scalar, PyTree, PyTree, PyTree, PyTree], PyTree],
    num_g_states: int,
    g0: PyTree,
) -> Callable[[Scalar, PyTree, PyTree, PyTree], PyTree]:
    @functools.wraps(f)
    def wrapper(t: Scalar, y: PyTree, u: PyTree, args: PyTree) -> PyTree:
        fy = y[..., :-num_g_states]
        gy = y[..., -num_g_states:]

        df_dt = f(t, fy, u, args)
        dg_dt = g(t, fy, gy, u, args)

        return jnp.concatenate((df_dt, dg_dt), axis=-1)

    def modify_initial_state(control: PyTree, t0: Scalar, y0: Array) -> Array:
        return jnp.concatenate((y0, g0), axis=-1)

    wrapper_fn = wrapper
    wrapper_fn._modify_initial_state = modify_initial_state

    return wrapper_fn


@eqx.filter_jit
def _eval_traj_stepping_nested(
    control: eqx.Module,
    ode: Callable[[Scalar, PyTree, PyTree, PyTree], PyTree],
    t1: float,
    y0: PyTree,
    step_dt: float,
):
    # This technique (essentially interrupting the solve at each control update)
    # doesn't seem that great, and gives issues on the backward pass
    # Manual stepping is probably more appropiate
    # However, this might be worth revisiting

    # num_steps = jnp.ceil(t1 / step_dt).astype(jnp.int_)
    num_steps = int(math.ceil(t1 / step_dt))

    def step_fn(
        diffeq_solver_state: Optional[PyTree],
        diffeq_controller_state: Optional[PyTree],
        diffeq_made_jump: Optional[Any],
        control_state: PyTree,
        y0: PyTree,
        t0: float,
        t1: float,
    ) -> Tuple[diffrax.Solution, PyTree, PyTree]:
        control_values, next_control_state = control(y0, control_state)

        next_diffeq_state = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(lambda t, y, args: ode(t, y, control_values, args)),
            solver=diffrax.Kvaerno5(),
            t0=t0,
            t1=t1,
            dt0=0.01,
            y0=y0,
            args=None,
            saveat=diffrax.SaveAt(
                t1=True, solver_state=True, controller_state=True, made_jump=True
            ),
            stepsize_controller=diffrax.PIDController(
                rtol=1e-5, atol=1e-5, pcoeff=0.3, icoeff=0.3
            ),
            solver_state=diffeq_solver_state,
            controller_state=diffeq_controller_state,
            made_jump=diffeq_made_jump,
        )

        return next_diffeq_state, next_control_state, control_values

    # For scanning over a bounded array, see
    # https://github.com/google/jax/issues/5642
    # https://github.com/google/jax/issues/5642

    # TODO: This errors out on the backwards pass
    # A fix might be to use eqx.internal.while_loop instead

    def scan_fn(
        carry: Tuple[diffrax.Solution, PyTree], x: float
    ) -> Tuple[Tuple[diffrax.Solution, PyTree], Tuple[PyTree, PyTree]]:
        diffeq_state, control_state = carry
        t1 = x

        next_diffeq_state, next_control_state, control_output = step_fn(
            diffeq_solver_state=diffeq_state.solver_state,
            diffeq_controller_state=diffeq_state.controller_state,
            diffeq_made_jump=diffeq_state.made_jump,
            control_state=control_state,
            y0=diffeq_state.ys[-1],
            t0=diffeq_state.ts[-1],
            t1=t1,
        )

        yt1 = next_diffeq_state.ys[-1]
        return (next_diffeq_state, next_control_state), (yt1, control_output)

    # Manually do first step
    # This can't be moved into scan, since the diffeq states are None, but scan
    # expects identical shapes during all iterations
    ts = jnp.linspace(0.0, t1, num=num_steps)

    init_diffeq_state, init_control_state, init_control_values = step_fn(
        diffeq_solver_state=None,
        diffeq_controller_state=None,
        diffeq_made_jump=None,
        control_state=control.main.initial_state,
        y0=y0,
        t0=ts[0],
        t1=ts[1],
    )

    # Scan over remaining steps
    _, (scan_ys, scan_cs) = jax.lax.scan(
        scan_fn, init=(init_diffeq_state, init_control_state), xs=ts[2:]
    )

    # Assemble solution
    y01 = jnp.stack((y0, init_diffeq_state.ys[-1]), axis=0)
    ys = jnp.concatenate((y01, scan_ys), axis=0)

    full_diffeq_solution = diffrax.Solution(
        t0=ts[0],
        t1=ts[-1],
        ts=ts,
        ys=ys,
        interpolation=None,
        stats=dict(),
        result=diffrax.RESULTS.successful,
        solver_state=None,
        controller_state=None,
        made_jump=None,
    )

    return full_diffeq_solution


# TODO: Adapt signature to match diffrax
@eqx.filter_jit
def diffeqsolve_drnn_controller(
    ode: Callable[[Scalar, PyTree, PyTree, PyTree], PyTree],
    solver: diffrax.AbstractSolver,
    stepsize_controller: diffrax.AbstractStepSizeController,
    control: eqx.Module,
    trajectory_t1: float,
    init_t0: float,
    init_dt0: float,
    init_y0: PyTree,
    step_dt: float,
    max_steps: int = 4096,
) -> Tuple[Array, Array, Array, Array]:
    # MAYBE TODO: Use internals from diffrax and just change adjoint.loop

    ## New version with manual stepping

    # Params
    buffer_len = max_steps
    # buffer_len = int(math.ceil(trajectory_t1 / step_dt))

    term = diffrax.ODETerm(lambda t, y, args: ode(t, y, args[0], args[1]))
    solver = stepsize_controller.wrap_solver(solver)

    # Initialize solver components
    init_control_state = control.main.initial_state
    init_control_value, _ = control(init_y0, init_control_state)

    init_t1, init_stepsize_controller_state = stepsize_controller.init(
        terms=term,
        t0=init_t0,
        t1=init_t0 + init_dt0,
        y0=init_y0,
        dt0=init_dt0,
        args=(init_control_value, None),
        func=solver.func(term, init_t0, init_y0, (init_control_value, None)),
        error_order=solver.error_order(term),
    )

    init_solver_state = solver.init(
        term, init_t1, trajectory_t1, init_y0, (init_control_value, None)
    )

    valid_buffer = jnp.zeros((buffer_len, 1))
    time_buffer = jnp.zeros((buffer_len, 1))
    solution_buffer = jnp.zeros((buffer_len, init_y0.shape[-1]))
    control_buffer = jnp.zeros((buffer_len, init_control_value.shape[-1]))

    valid_buffer = valid_buffer.at[0].set(True)
    time_buffer = time_buffer.at[0].set(init_t0)
    solution_buffer = solution_buffer.at[0].set(init_y0)
    control_buffer = control_buffer.at[0].set(init_control_value)

    # Integrate ODE
    class Carry(eqx.Module):
        t0: float
        t1: float
        y0: float
        c0: float
        control_steps: int
        solver_state: PyTree
        stepsize_controller_state: PyTree
        control_state: PyTree
        made_jump: Any
        valid_buffer: Array
        time_buffer: Array
        solution_buffer: Array
        control_buffer: Array

    def step_fn(carry: Carry) -> Carry:
        # Stepping algorithm overview
        # We are guaranteed to start the first iteration with t0 inside of the
        # controller step boundary, and a valid c0
        # At every iteration, we check if the current upper border of the integration
        # interval t1 goes past the controller step boundary. If so, we clamp it to the
        # boundary. Then, at the next step, t0 will either be the previous t1 or
        # nextafter(t1). Hence, we can always simply check if t0 is past the step
        # boundary, to decide if we need to step the controller. If so, we update
        # the control values and calculate new controller step boundaries.

        # Get controller step boundary
        ct1 = init_t0 + step_dt * (carry.control_steps + 1)

        # Is the start of the current integration interval at the controller step
        # boundary?
        # Note: Here, >= is used, because t0 could also be nextafter(ct1), so == could
        # fail in certain edge cases.
        crossed_boundary = carry.t0 >= ct1

        # If so, we increase the number of controller steps...
        next_control_steps = jnp.where(
            crossed_boundary, carry.control_steps + 1, carry.control_steps
        )

        # ... and recalculate the controller step boundary
        ct0 = init_t0 + step_dt * next_control_steps
        ct1 = init_t0 + step_dt * (next_control_steps + 1)

        # Then, with the correct step boundary, we clamp the upper border of the
        # integration interval to not go past the new boundary
        t1 = jnp.where(carry.t1 > ct1, ct1, carry.t1)

        # Get the current control values, either from the controller if the boundary
        # was crossed or the previous values if we are still in the same controller step
        c0, next_control_state = jax.lax.cond(
            crossed_boundary,
            lambda y0, c0, state: control(y0, state),
            lambda y0, c0, state: (c0, state),
            carry.y0,
            carry.c0,
            carry.control_state,
        )

        def _update_buffers(
            buffer_idx: int,
            time_values: Array,
            solution_values: Array,
            control_values: Array,
            valid_buffer: Array,
            time_buffer: Array,
            solution_buffer: Array,
            control_buffer: Array,
        ) -> Tuple[Array, Array, Array]:
            return (
                valid_buffer.at[buffer_idx].set(True),
                time_buffer.at[buffer_idx].set(time_values),
                solution_buffer.at[buffer_idx].set(solution_values),
                control_buffer.at[buffer_idx].set(control_values),
            )

        def _identity_buffers(
            buffer_idx: int,
            time_values: Array,
            solution_values: Array,
            control_values: Array,
            valid_buffer: Array,
            time_buffer: Array,
            solution_buffer: Array,
            control_buffer: Array,
        ) -> Tuple[Array, Array, Array]:
            return (valid_buffer, time_buffer, solution_buffer, control_buffer)

        # Write solution state into buffers, if boundary was crossed
        buffer_idx = next_control_steps
        valid_buffer, time_buffer, solution_buffer, control_buffer = jax.lax.cond(
            crossed_boundary,
            _update_buffers,
            _identity_buffers,
            buffer_idx,
            carry.t0,
            carry.y0,
            c0,
            carry.valid_buffer,
            carry.time_buffer,
            carry.solution_buffer,
            carry.control_buffer,
        )

        # Attempt a step
        y1, local_error_est, dense, next_solver_state, result = solver.step(
            terms=term,
            t0=carry.t0,
            t1=t1,
            y0=carry.y0,
            args=(c0, None),
            solver_state=carry.solver_state,
            made_jump=jnp.logical_or(crossed_boundary, carry.made_jump),
        )

        (
            step_accepted,
            next_t0,
            next_t1,
            made_jump,
            next_stepsize_controller_state,
            result,
        ) = stepsize_controller.adapt_step_size(
            t0=carry.t0,
            t1=t1,
            y0=carry.y0,
            y1_candidate=y1,
            args=(c0, None),
            y_error=local_error_est,
            error_order=solver.order(term),
            controller_state=carry.stepsize_controller_state,
        )

        return Carry(
            t0=next_t0,
            t1=next_t1,
            y0=y1,
            c0=c0,
            control_steps=next_control_steps,
            solver_state=next_solver_state,
            stepsize_controller_state=next_stepsize_controller_state,
            control_state=next_control_state,
            made_jump=made_jump,
            valid_buffer=valid_buffer,
            time_buffer=time_buffer,
            solution_buffer=solution_buffer,
            control_buffer=control_buffer,
        )

    def cond_fn(carry: Carry) -> bool:
        return carry.t0 < trajectory_t1

    last = eqx.internal.while_loop(
        cond_fun=cond_fn,
        body_fun=step_fn,
        init_val=Carry(
            t0=init_t0,
            t1=init_t1,
            y0=init_y0,
            c0=init_control_value,
            control_steps=jnp.int_(0),
            solver_state=init_solver_state,
            stepsize_controller_state=init_stepsize_controller_state,
            control_state=init_control_state,
            made_jump=jnp.bool_(False),
            valid_buffer=valid_buffer,
            time_buffer=time_buffer,
            solution_buffer=solution_buffer,
            control_buffer=control_buffer,
        ),
        max_steps=4096,
        buffers=lambda carry: (
            carry.valid_buffer,
            carry.time_buffer,
            carry.solution_buffer,
            carry.control_buffer,
        ),
        kind="checkpointed",
    )

    return (
        last.valid_buffer,
        last.time_buffer,
        last.solution_buffer,
        last.control_buffer,
    )
