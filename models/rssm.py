from typing import Optional, List

import jax
import jax.numpy as jnp

import flax.linen as nn

from neuroblast.models.common import MLP


class RSSM(nn.Module):
    rng: jnp.ndarray
    action_dim: int
    belief_dim: int = 200
    embedding_dim: int = 1024
    hidden_dim: int = 200
    state_dim: int = 100
    min_std_dev = 0.1

    def setup(self):
        self.fc_embed_state_action = MLP((self.belief_dim,), activate_final=True)
        self.rnn = nn.GRUCell()
        self.fc_embed_belief_prior = MLP((self.hidden_dim,), activate_final=True)
        self.fc_state_prior = MLP((2 * self.state_dim,), activate_final=True)
        self.fc_embed_belief_posterior = MLP((self.hidden_dim,), activate_final=True)
        self.fc_state_posterior = MLP((2 * self.state_dim,), activate_final=True)

    def __call__(self,
                 prev_state: jnp.ndarray,
                 prev_action: jnp.ndarray,
                 prev_belief: jnp.ndarray,
                 observation: Optional[jnp.ndarray] = None,
                 nonterminals: Optional[jnp.ndarray] = None) -> List[jnp.ndarray]:
        assert prev_state.shape[0] == 1

        # mask terminal states
        if nonterminals is not None:
            if prev_state.ndim == 2:
                # T * B * H = 1 X batch X state_dim
                prev_state = prev_state * jnp.tile(nonterminals, (1, prev_state.shape[1]))
            else:
                prev_state = prev_state * jnp.tile(nonterminals, (1, 1, prev_state.shape[2]))

        # Compute belief (deterministic hidden state)
        hidden = self.fc_embed_state_action(jnp.concatenate([prev_state, prev_action], axis=-1))
        _, new_belief = self.rnn(hidden, prev_belief)

        # Compute state prior by applying transition dynamics
        hidden = self.fc_embed_belief_prior(new_belief)
        prior_mean, prior_std_dev = jnp.split(self.fc_state_prior(hidden), 2, axis=-1)
        prior_std_dev = nn.softplus(prior_std_dev) + self.min_std_dev
        prior_state = prior_mean + prior_std_dev * jax.random.normal(self.rng, prior_mean.shape)

        results = [new_belief, prior_state, prior_mean, prior_std_dev]

        if observation is not None:
            # Compute state posterior by applying transition dynamics and using current observation
            hidden = self.fc_embed_belief_posterior(
                jnp.concatenate([new_belief, observation], axis=-1))
            posterior_mean, posterior_std_dev = jnp.split(self.fc_state_posterior(hidden), 2, axis=-1)
            posterior_std_dev = nn.softplus(posterior_std_dev) + self.min_std_dev
            posterior_state = posterior_mean + posterior_std_dev * jax.random.normal(self.rng, posterior_mean.shape)

            results += [posterior_state, posterior_mean, posterior_std_dev]
        else:
            results += [None, None, None]

        return results
