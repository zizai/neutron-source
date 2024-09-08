"""Implementations of algorithms for continuous control."""

from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from models.common import MLP


class Critic(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self,
                 obs: jnp.ndarray,
                 actions: jnp.ndarray) -> jnp.ndarray:
        assert obs.shape[-1] == 2
        k = self.param('k', jax.nn.initializers.normal(1.),(2, 128))
        k = jax.lax.stop_gradient(k)
        k = 2 * jnp.pi * k
        x = jnp.concatenate([jnp.sin(obs @ k), jnp.cos(obs @ k)], -1)

        inputs = jnp.concatenate([x, actions], -1)
        # out = MLP((*self.hidden_dims, 4), activate_final=True)(inputs)
        # out = out.reshape(*(out.shape[:-2] + (-1,)))
        # critic = MLP((512, 256, 1))(out)
        # critic = jnp.squeeze(critic, -1)
        critic = MLP((*self.hidden_dims, 1))(inputs)
        critic = jnp.mean(jnp.squeeze(critic, -1), -1)
        return critic


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        critic1 = Critic(self.hidden_dims)(observations, actions)
        critic2 = Critic(self.hidden_dims)(observations, actions)
        return critic1, critic2
