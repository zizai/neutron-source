from functools import partial
from typing import Sequence, Tuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from models.common import MLP, Params, PRNGKey

LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0


class NormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    state_dependent_std: bool = True

    @nn.compact
    def __call__(self,
                 obs: jnp.ndarray,
                 temperature: float = 1.0) -> distrax.Distribution:
        assert obs.shape[-1] == 2
        k = self.param('k', jax.nn.initializers.normal(1.),(2, 128))
        k = jax.lax.stop_gradient(k)
        k = 2 * jnp.pi * k
        x = jnp.concatenate([jnp.sin(obs @ k), jnp.cos(obs @ k)], -1)

        outputs = MLP(self.hidden_dims, activate_final=True)(x)

        means = nn.Dense(self.action_dim)(outputs)

        if self.state_dependent_std:
            log_stds = nn.Dense(self.action_dim)(outputs)
        else:
            log_stds = self.param('log_stds', nn.initializers.zeros,
                                  (self.action_dim, ))

        log_stds = jnp.clip(log_stds, LOG_STD_MIN, LOG_STD_MAX)

        base_dist = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        return distrax.Transformed(distribution=base_dist,
                                   bijector=distrax.Block(distrax.Tanh(), 1))


class NormalTanhMixturePolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    num_components: int = 4

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0) -> distrax.Distribution:
        outputs = MLP(self.hidden_dims, activate_final=True)(observations)

        logits = nn.Dense(self.action_dim * self.num_components)(outputs)
        means = nn.Dense(self.action_dim * self.num_components,
                         bias_init=nn.initializers.normal(stddev=1.0))(outputs)
        log_stds = nn.Dense(self.action_dim * self.num_components)(outputs)

        shape = list(observations.shape[:-1]) + [-1, self.num_components]
        logits = jnp.reshape(logits, shape)
        mu = jnp.reshape(means, shape)
        log_stds = jnp.reshape(log_stds, shape)

        log_stds = jnp.clip(log_stds, LOG_STD_MIN, LOG_STD_MAX)

        components_distribution = distrax.Normal(loc=mu,
                                                 scale=jnp.exp(log_stds) *
                                                 temperature)

        base_dist = distrax.MixtureSameFamily(
            mixture_distribution=distrax.Categorical(logits=logits),
            components_distribution=components_distribution)

        dist = distrax.Transformed(distribution=base_dist,
                                   bijector=distrax.Tanh())

        return distrax.Independent(dist, 1)


class BinomialPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int

    @nn.compact
    def __call__(self, obs, temp=1.0):
        outputs = MLP(self.hidden_dims, activate_final=True)(obs)

        logits = nn.Dense(self.action_dim)(outputs)
        dist = distrax.Bernoulli(logits=logits)
        return distrax.Independent(dist, 1)


def sample_action(rng: PRNGKey,
                  actor_def: nn.Module,
                  actor_params: Params,
                  obs: np.ndarray,
                  temp: float) -> Tuple[PRNGKey, jnp.ndarray]:
    dist = actor_def.apply({'params': actor_params}, obs, temp)
    rng, key = jax.random.split(rng)
    if isinstance(actor_def, BinomialPolicy):
        return rng, dist.sample(seed=key)
    else:
        return rng, dist.sample(seed=key) > 0


def take_action(actor_def: nn.Module,
                actor_params: Params,
                obs: np.ndarray,
                temp: float) -> Tuple[PRNGKey, jnp.ndarray]:
    dist = actor_def.apply({'params': actor_params}, obs, temp)
    if isinstance(actor_def, BinomialPolicy):
        return dist.distribution.mean() > 0.5
    else:
        return dist.distribution.mean() > 0
