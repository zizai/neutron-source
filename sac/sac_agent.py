"""Implementations of algorithms for continuous control."""
from functools import partial
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxlib.xla_extension import Device

from trainer.dataset import Batch
from models import policies, critic_net
from models.common import ActorCriticTemp, InfoDict, Model
from sac import temperature, critic, actor


def update_sac(sac: ActorCriticTemp,
               batch: Batch,
               discount: float,
               tau: float,
               target_entropy: float,
               update_target: bool) -> Tuple[ActorCriticTemp, InfoDict]:

    sac, critic_info = critic.update(sac, batch, discount, soft_critic=True)
    if update_target:
        sac = critic.target_update(sac, tau)

    sac, actor_info = actor.update(sac, batch)
    sac, alpha_info = temperature.update(sac, actor_info['entropy'], target_entropy)
    # alpha_info = {}

    return sac, {**critic_info, **actor_info, **alpha_info}


class SACAgent(object):
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 tau: float = 0.005,
                 target_update_period: int = 1,
                 target_entropy: Optional[float] = None,
                 init_temperature: float = 1.0,
                 device: Device = None):

        action_dim = actions.shape[-1]

        if target_entropy is None:
            self.target_entropy = -action_dim / 2
        else:
            self.target_entropy = target_entropy

        self.tau = tau
        self.target_update_period = target_update_period
        self.discount = discount
        self.device = device

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        actor_def = policies.BinomialPolicy(hidden_dims, action_dim)
        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=optax.adam(learning_rate=actor_lr))

        critic_def = critic_net.DoubleCritic(hidden_dims)
        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=optax.adam(learning_rate=critic_lr))
        target_critic = Model.create(critic_def,
                                     inputs=[critic_key, observations, actions])

        temp = Model.create(temperature.Temperature(init_temperature),
                            inputs=[temp_key],
                            tx=optax.adam(learning_rate=temp_lr))

        self.sac = ActorCriticTemp(actor=actor,
                                   critic=critic,
                                   target_critic=target_critic,
                                   temp=temp,
                                   rng=rng)
        self.step = 1

    def sample_action(self, obs) -> jnp.ndarray:
        if self.device is None:
            _sample_action = jax.jit(policies.sample_action, static_argnums=(1,))
        else:
            _sample_action = jax.jit(policies.sample_action, static_argnums=(1,), device=self.device)

        rng, action = _sample_action(self.sac.rng,
                                     self.sac.actor.apply_fn,
                                     self.sac.actor.params,
                                     obs)
        self.sac = self.sac.replace(rng=rng)
        return action

    def take_action(self, obs: np.ndarray) -> jnp.ndarray:
        if self.device is None:
            _take_action = jax.jit(policies.take_action, static_argnums=(0,))
        else:
            _take_action = jax.jit(policies.take_action, static_argnums=(0,), device=self.device)

        action = _take_action(self.sac.actor.apply_fn, self.sac.actor.params, obs)
        return action

    def learn_on_batch(self, batch: Batch) -> InfoDict:
        if self.device is None:
            _update_sac = jax.jit(update_sac, static_argnums=(2, 3, 4, 5))
        else:
            _update_sac = jax.jit(update_sac, static_argnums=(2, 3, 4, 5), device=self.device)

        self.sac, info = _update_sac(self.sac,
                                     batch,
                                     self.discount,
                                     self.tau,
                                     self.target_entropy,
                                     self.step % self.target_update_period == 0)

        self.step += 1
        return info
