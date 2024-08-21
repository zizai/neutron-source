import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import partial
from jax.experimental import stax, optimizers
from jax.experimental.ode import odeint


# unconstrained equation of motion
def unconstrained_eom(model, state, t=None):
    q, q_t = jnp.split(state, 2)
    return model(q, q_t)


# lagrangian equation of motion
def lagrangian_eom(lagrangian, state, t=None):
    q, q_t = jnp.split(state, 2, axis=-1)
    # Note: the following line assumes q is an angle. Delete it for problems other than double pendulum.
    q = q % (2 * jnp.pi)
    q_tt = (jnp.linalg.pinv(jax.hessian(lagrangian, 1)(q, q_t))
            @ (jax.grad(lagrangian, 0)(q, q_t)
               - jax.jacobian(jax.jacobian(lagrangian, 1), 0)(q, q_t) @ q_t))
    # dt = 1e-1
    # return dt * jnp.concatenate([q_t, q_tt])
    return jnp.concatenate([q_t, q_tt], axis=-1)


def raw_lagrangian_eom(lagrangian, state, t=None):
    q, q_t = jnp.split(state, 2, axis=-1)
    q = q % (2 * jnp.pi)
    q_tt = (jnp.linalg.pinv(jax.hessian(lagrangian, 1)(q, q_t))
            @ (jax.grad(lagrangian, 0)(q, q_t)
               - jax.jacobian(jax.jacobian(lagrangian, 1), 0)(q, q_t) @ q_t))
    return jnp.concatenate([q_t, q_tt])


def lagrangian_eom_rk4(lagrangian, state, n_updates, Dt=1e-1, t=None):
    @jax.jit
    def cur_fnc(state):
        q, q_t = jnp.split(state, 2, axis=-1)
        q = q % (2 * jnp.pi)
        q_tt = (jnp.linalg.pinv(jax.hessian(lagrangian, 1)(q, q_t))
                @ (jax.grad(lagrangian, 0)(q, q_t)
                   - jax.jacobian(jax.jacobian(lagrangian, 1), 0)(q, q_t) @ q_t))
        return jnp.concatenate([q_t, q_tt], axis=-1)

    @jax.jit
    def get_update(update):
        dt = Dt / n_updates
        cstate = state + update
        k1 = dt * cur_fnc(cstate)
        k2 = dt * cur_fnc(cstate + k1 / 2)
        k3 = dt * cur_fnc(cstate + k2 / 2)
        k4 = dt * cur_fnc(cstate + k3)
        return update + 1.0 / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    update = 0
    for _ in range(n_updates):
        update = get_update(update)
    return update


def solve_dynamics(dynamics_fn, initial_state, is_lagrangian=True, **kwargs):
    eom = lagrangian_eom if is_lagrangian else unconstrained_eom

    # We currently run odeint on CPUs only, because its cost is dominated by
    # control flow, which is slow on GPUs.
    @partial(jax.jit, backend='cpu')
    def f(initial_state):
        return odeint(partial(eom, dynamics_fn), initial_state, **kwargs)

    return f(initial_state)


def custom_init(init_params, seed=0):
    """Do an optimized LNN initialization for a simple uniform-width MLP"""
    import numpy as np
    new_params = []
    rng = jax.random.PRNGKey(seed)
    i = 0
    number_layers = len([0 for l1 in init_params if len(l1) != 0])
    for l1 in init_params:
        if (len(l1)) == 0: new_params.append(()); continue
        new_l1 = []
        for l2 in l1:
            if len(l2.shape) == 1:
                # Zero init biases
                new_l1.append(jnp.zeros_like(l2))
            else:
                n = max(l2.shape)
                first = int(i == 0)
                last = int(i == number_layers - 1)
                mid = int((i != 0) * (i != number_layers - 1))
                mid *= i

                std = 1.0 / np.sqrt(n)
                std *= 2.2 * first + 0.58 * mid + n * last

                if std == 0:
                    raise NotImplementedError("Wrong dimensions for MLP")

                new_l1.append(jax.random.normal(rng, l2.shape) * std)
                rng += 1
                i += 1

        new_params.append(new_l1)

    return new_params


def mlp(hidden_dim, output_dim):
    return stax.serial(
        stax.Dense(hidden_dim),
        stax.Softplus,
        stax.Dense(hidden_dim),
        stax.Softplus,
        stax.Dense(output_dim),
    )


def pixel_encoder(args):
    return stax.serial(
        stax.Dense(args.ae_hidden_dim),
        stax.Softplus,
        stax.Dense(args.ae_latent_dim),
    )


def pixel_decoder(args):
    return stax.serial(
        stax.Dense(args.ae_hidden_dim),
        stax.Softplus,
        stax.Dense(args.ae_input_dim),
    )


def wrap_coords(state):
    # wrap generalized coordinates to [-pi, pi]
    q_dim = int(state.shape[-1] / 2)
    return jnp.concatenate([(state[:q_dim] + jnp.pi) % (2 * jnp.pi) - jnp.pi, state[q_dim:]])


def rk4_step(f, x, t, h):
    # one step of Runge-Kutta integration
    k1 = h * f(x, t)
    k2 = h * f(x + k1/2, t + h/2)
    k3 = h * f(x + k2/2, t + h/2)
    k4 = h * f(x + k3, t + h)
    return x + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)


def radial2cartesian(t1, t2, l1, l2):
    # Convert from radial to Cartesian coordinates.
    x1 = l1 * jnp.sin(t1)
    y1 = -l1 * jnp.cos(t1)
    x2 = x1 + l2 * jnp.sin(t2)
    y2 = y1 - l2 * jnp.cos(t2)
    return x1, y1, x2, y2


def learned_dynamics(nn_forward_fn, params):
    # replace the lagrangian with a parameteric model
    def dynamics(q, q_t):
        state = jnp.concatenate([q, q_t], axis=-1)
        return jnp.squeeze(nn_forward_fn(params, state), axis=-1)
    return dynamics


@partial(jax.jit, static_argnums=(0,))
def gln_loss(nn_forward_fn, params, batch):
    state, targets = batch
    preds = jax.vmap(partial(lagrangian_eom, learned_dynamics(nn_forward_fn, params)))(state)
    return jnp.mean((preds - targets) ** 2)


@partial(jax.jit, static_argnums=(0,))
def baseline_loss(nn_forward_fn, params, batch):
    state, targets = batch
    preds = jax.vmap(partial(unconstrained_eom, learned_dynamics(nn_forward_fn, params)))(state)
    return jnp.mean((preds - targets) ** 2)
