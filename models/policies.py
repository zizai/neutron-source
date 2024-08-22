from functools import partial
from typing import Sequence, Tuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from models.common import MLP, Params, PRNGKey, default_init

LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0


class NormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    state_dependent_std: bool = True

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0) -> distrax.Distribution:
        outputs = MLP(self.hidden_dims, activate_final=True)(observations)

        means = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)

        if self.state_dependent_std:
            log_stds = nn.Dense(self.action_dim,
                                kernel_init=default_init())(outputs)
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
    num_components: int = 5

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0) -> distrax.Distribution:
        outputs = MLP(self.hidden_dims, activate_final=True)(observations)

        logits = nn.Dense(self.action_dim * self.num_components,
                          kernel_init=default_init())(outputs)
        means = nn.Dense(self.action_dim * self.num_components,
                         kernel_init=default_init(),
                         bias_init=nn.initializers.normal(stddev=1.0))(outputs)
        log_stds = nn.Dense(self.action_dim * self.num_components,
                            kernel_init=default_init())(outputs)

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


@partial(jax.jit, static_argnums=1)
def sample_actions(rng: PRNGKey,
                   actor_def: nn.Module,
                   actor_params: Params,
                   observations: np.ndarray,
                   temperature: float = 1.0) -> Tuple[PRNGKey, jnp.ndarray]:
    dist = actor_def.apply({'params': actor_params}, observations, temperature)
    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=key)
