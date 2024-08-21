from typing import Optional, List

import jax
import jax.numpy as jnp

import flax.linen as nn

from neuroblast.models.common import MLP
from neuroblast.models.world_model import SymbolicEncoder, SymbolicDecoder


def reparameterize(rng, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = jax.random.normal(rng, logvar.shape)
    return mean + eps * std


class TransitionModel(nn.Module):
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

    # Operates over (previous) state, (previous) actions, (previous) belief, (previous) nonterminals (mask), and (current) observations
    # Diagram of expected inputs and outputs for T = 5 (-x- signifying beginning of output belief/state that gets sliced off):
    # t :  0  1  2  3  4  5
    # o :    -X--X--X--X--X-
    # a : -X--X--X--X--X-
    # n : -X--X--X--X--X-
    # pb: -X-
    # ps: -X-
    # b : -x--X--X--X--X--X-
    # s : -x--X--X--X--X--X-
    def __call__(self,
                 prev_state: jnp.ndarray,
                 actions: jnp.ndarray,
                 prev_belief: jnp.ndarray,
                 observations: Optional[jnp.ndarray] = None,
                 nonterminals: Optional[jnp.ndarray] = None) -> List[jnp.ndarray]:
        # Create lists for hidden states
        T = actions.shape[0]
        beliefs, prior_states, prior_means, prior_std_devs = [], [], [], []
        posterior_states, posterior_means, posterior_std_devs = [], [], []
        beliefs.append(prev_belief)
        prior_states.append(prev_state)
        posterior_states.append(prev_state)

        # Loop over time sequence
        for t in range(T):
            # Select appropriate previous state
            _state = prior_states[t] if observations is None else posterior_states[t]
            # Mask if previous transition was terminal
            _state = _state if nonterminals is None else _state * nonterminals[t]

            # Compute belief (deterministic hidden state)
            current_action = jnp.expand_dims(actions[t], 0)
            hidden = self.fc_embed_state_action(jnp.concatenate([_state, current_action], axis=-1))
            _, new_belief = self.rnn(hidden, beliefs[t])

            # Compute state prior by applying transition dynamics
            hidden = self.fc_embed_belief_prior(new_belief)
            new_prior_mean, new_prior_std_dev = jnp.split(self.fc_state_prior(hidden), 2, axis=-1)
            new_prior_std_dev = nn.softplus(new_prior_std_dev) + self.min_std_dev
            new_prior_state = new_prior_mean + new_prior_std_dev * jax.random.normal(self.rng, new_prior_mean.shape)

            beliefs.append(new_belief)
            prior_states.append(new_prior_state)
            prior_means.append(new_prior_mean)
            prior_std_devs.append(new_prior_std_dev)

            if observations is not None:
                # Compute state posterior by applying transition dynamics and using current observation
                current_obs = jnp.expand_dims(observations[t], 0)
                hidden = self.fc_embed_belief_posterior(
                    jnp.concatenate([new_belief, current_obs], axis=-1))
                new_posterior_mean, new_posterior_std_dev = jnp.split(self.fc_state_posterior(hidden), 2, axis=-1)
                new_posterior_std_dev = nn.softplus(new_posterior_std_dev) + self.min_std_dev
                new_posterior_state = new_posterior_mean + new_posterior_std_dev * \
                                      jax.random.normal(self.rng, new_posterior_mean.shape)

                posterior_states.append(new_posterior_state)
                posterior_means.append(new_posterior_mean)
                posterior_std_devs.append(new_posterior_std_dev)

        # Return new hidden states
        hidden = [jnp.concatenate(beliefs[1:], axis=0), jnp.concatenate(prior_states[1:], axis=0),
                  jnp.concatenate(prior_means, axis=0), jnp.concatenate(prior_std_devs, axis=0)]
        if observations is not None:
            hidden += [jnp.concatenate(posterior_states[1:], axis=0), jnp.concatenate(posterior_means, axis=0),
                       jnp.concatenate(posterior_std_devs, axis=0)]
        else:
            hidden += [None, None, None]
        return hidden


class RewardModel(nn.Module):
    belief_dim: int
    hidden_dim: int
    state_dim: int

    @nn.compact
    def __call__(self, belief, state):
        layers = MLP((self.hidden_dim, self.hidden_dim, 1), activate_final=False)
        reward = layers(jnp.concatenate([belief, state], axis=-1)).squeeze(axis=-1)
        return reward


class PlaNet(nn.Module):
    rng: jnp.ndarray
    observation_dim: int
    action_dim: int
    belief_dim: int = 200
    embedding_dim: int = 1024
    hidden_dim: int = 200
    state_dim: int = 100
    plan_horizon: int = 10
    plan_iters: int = 10
    candidates: int = 1000
    top_candidates: int = 100
    min_action: float = -1.0
    max_action: float = 1.0

    def setup(self):
        self.encoder = SymbolicEncoder((self.embedding_dim, self.embedding_dim))
        self.decoder = SymbolicDecoder((self.embedding_dim, self.observation_dim))

        _, transition_key = jax.random.split(self.rng)
        self.transition_model = TransitionModel(rng=transition_key,
                                                action_dim=self.action_dim,
                                                belief_dim=self.belief_dim,
                                                embedding_dim=self.embedding_dim,
                                                hidden_dim=self.hidden_dim,
                                                state_dim=self.state_dim)
        self.reward_model = RewardModel(belief_dim=self.belief_dim,
                                        hidden_dim=self.hidden_dim,
                                        state_dim=self.state_dim)

    def __call__(self,
                 observation: jnp.ndarray,
                 prev_action: Optional[jnp.ndarray] = None,
                 prev_belief: Optional[jnp.ndarray] = None,
                 prev_state: Optional[jnp.ndarray] = None,
                 masks: Optional[jnp.ndarray] = None,
                 plan_mode: bool = False):
        assert observation.ndim == 2
        B = observation.shape[0]
        prev_action = jnp.zeros((B, self.action_dim)) if prev_action is None else prev_action
        prev_belief = jnp.zeros((1, self.belief_dim)) if prev_belief is None else prev_belief
        prev_state = jnp.zeros((1, self.state_dim)) if prev_state is None else prev_state

        if plan_mode:
            H, Z = prev_belief.shape[-1], prev_state.shape[-1]
            observation, prev_action = observation.reshape((B, -1)), prev_action.reshape((B, -1))

            embedding = self.encoder(observation)
            belief, _, _, _, posterior_state, posterior_mean, posterior_std_dev = self.transition_model(
                prev_state, prev_action, prev_belief, embedding)

            # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
            _belief = jnp.tile(belief.reshape(1, B, 1, H), (1, 1, self.candidates, 1))
            _state = jnp.tile(posterior_state.reshape(1, B, 1, Z), (1, 1, self.candidates, 1))
            action_mean = jnp.zeros((self.plan_horizon, B, 1, self.action_dim))
            action_std_dev = jnp.ones((self.plan_horizon, B, 1, self.action_dim))
            for _ in range(self.plan_iters):
                # Evaluate J action sequences from the current belief
                # over entire sequence at once, batched over particles
                actions = action_mean + action_std_dev * jax.random.normal(self.rng, (
                self.plan_horizon, B, self.candidates, self.action_dim))
                # Clip action range
                actions = actions.clip(self.min_action, self.max_action)

                # Sample next states
                beliefs, states, _, _, _, _, _ = self.transition_model(_state, actions, _belief)

                # Calculate expected returns (technically sum of rewards over planning horizon)
                r_pred = self.reward_model(beliefs, states)
                r_pred = r_pred.sum(axis=0).reshape(B, self.candidates)

                # Re-fit belief to the K best action sequences
                _, topk = jax.lax.top_k(r_pred, self.top_candidates)
                # Sample actions (time x (batch x candidates) x actions)
                # Fix indices for unrolled actions
                actions = actions.reshape(self.plan_horizon, B * self.candidates, self.action_dim)
                topk += jnp.expand_dims(self.candidates * jnp.array(range(0, B)), 1)
                best_actions = actions[:, topk.reshape(-1)].reshape(
                    self.plan_horizon, B, self.top_candidates, self.action_dim)

                # Update belief with new means and standard deviations
                action_mean = best_actions.mean(axis=2, keepdims=True)
                action_std_dev = best_actions.std(axis=2, keepdims=True)

            return belief, None, (posterior_state, posterior_mean, posterior_std_dev), None, None, action_mean[0].squeeze(axis=1)
        else:
            masks = jnp.ones((B,)) if masks is None else masks

            # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
            beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = self.transition_model(
                prev_state, prev_action, prev_belief, self.encoder(observation), masks)

            o_pred = self.decoder(jnp.concatenate([beliefs, posterior_states], axis=-1))
            r_pred = self.reward_model(beliefs, posterior_states)
            return beliefs, (prior_states, prior_means, prior_std_devs), (posterior_states, posterior_means, posterior_std_devs), o_pred, r_pred, None
