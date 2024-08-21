from typing import Optional, List, NamedTuple

import distrax
import jax
import jax.numpy as jnp
import jax.ops as jops

import flax.linen as nn

from neuroblast.models.common import MLP
from neuroblast.models.critic_net import Critic
from neuroblast.models.policies import NormalTanhPolicy
from neuroblast.models.rssm import RSSM
from neuroblast.models.world_model import SymbolicEncoder, SymbolicDecoder, VisualEncoder, VisualDecoder
from neuroblast.sim.constants import CompositeObservation


class RewardModel(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, belief, state):
        layers = MLP((self.hidden_dim, self.hidden_dim, 1), activate_final=False)
        reward = layers(jnp.concatenate([belief, state], axis=-1)).squeeze(axis=-1)
        return reward


class Dreamer(nn.Module):
    rng: jnp.ndarray
    observation_dim: int
    action_dim: int
    belief_dim: int = 512
    embedding_dim: int = 256
    hidden_dim: int = 256
    state_dim: int = 256

    def setup(self):
        self.visual_encoder = VisualEncoder()
        self.visual_decoder = VisualDecoder()
        self.symbolic_encoder = SymbolicEncoder((self.embedding_dim, self.embedding_dim))
        self.symbolic_decoder = SymbolicDecoder((self.embedding_dim, self.observation_dim))

        _, transition_key = jax.random.split(self.rng)
        self.transition_model = RSSM(rng=transition_key,
                                     action_dim=self.action_dim,
                                     belief_dim=self.belief_dim,
                                     embedding_dim=self.embedding_dim * 2,
                                     hidden_dim=self.hidden_dim,
                                     state_dim=self.state_dim)

        self.reward_model = MLP((self.hidden_dim, self.hidden_dim, 1))
        self.state_to_embedding = MLP((self.embedding_dim * 2,), activate_final=True)

    def __call__(self,
                 observation: CompositeObservation,
                 prev_action: Optional[jnp.ndarray] = None,
                 prev_belief: Optional[jnp.ndarray] = None,
                 prev_state: Optional[jnp.ndarray] = None,
                 masks: Optional[jnp.ndarray] = None,
                 temperature: float = 1.0,
                 training: bool = True):
        if observation is None:
            a_tm1 = jnp.expand_dims(prev_action, axis=0)
            h_tm1 = jnp.expand_dims(prev_belief, axis=0)
            s_tm1 = jnp.expand_dims(prev_state, axis=0)
            h_t, ps_t, _, _, _, _, _ = self.transition_model(
                s_tm1, a_tm1, h_tm1)

            h_t = h_t.squeeze(0)
            ps_t = ps_t.squeeze(0)
            r_pred_t = self.reward_model(ps_t).squeeze(-1)

            return h_t, (ps_t, _, _), None, None, r_pred_t

        so_t = observation.symbolic
        vo_t = observation.visual

        if so_t.ndim == 1:
            B = 1
            so_t = jnp.expand_dims(so_t, 0)
            vo_t = jnp.expand_dims(vo_t, 0)
        else:
            B = so_t.shape[0]

        if not training:
            a_tm1 = jnp.zeros((B, self.action_dim)) if prev_action is None else prev_action
            h_tm1 = jnp.zeros((B, self.belief_dim)) if prev_belief is None else prev_belief
            s_tm1 = jnp.zeros((B, self.state_dim)) if prev_state is None else prev_state

            sym_emb = self.symbolic_encoder(so_t).reshape(B, -1)
            vis_emb = self.visual_encoder(vo_t, training=training).reshape(B, -1)
            embedding = jnp.concatenate([sym_emb, vis_emb], axis=-1)

            a_tm1 = jnp.expand_dims(a_tm1, axis=0)
            h_tm1 = jnp.expand_dims(h_tm1, axis=0)
            s_tm1 = jnp.expand_dims(s_tm1, axis=0)
            embedding = jnp.expand_dims(embedding, axis=0)
            results = self.transition_model(s_tm1, a_tm1, h_tm1, embedding, masks)

            h_t, ps, p_mu, p_sigma, qs, q_mu, q_sigma = [r.squeeze(0) for r in results]

            r_pred = self.reward_model(qs).squeeze(-1)

            return h_t, None, (qs, q_mu, q_sigma), None, r_pred
        else:
            sym_emb = self.symbolic_encoder(so_t).reshape(B, -1)
            vis_emb = self.visual_encoder(vo_t, training=training).reshape((B, -1))
            embedding = jnp.concatenate([sym_emb, vis_emb], axis=-1)

            a_tm1 = jnp.expand_dims(prev_action, axis=0)
            h_tm1 = jnp.expand_dims(prev_belief, axis=0)
            s_tm1 = jnp.expand_dims(prev_state, axis=0)
            embedding = jnp.expand_dims(embedding, axis=0)
            masks = jnp.ones((1, B, 1)) if masks is None else masks.reshape((1, B, 1))
            results = self.transition_model(s_tm1, a_tm1, h_tm1, embedding, masks)

            h_t, ps, p_mu, p_sigma, qs, q_mu, q_sigma = [r.squeeze(0) for r in results]

            embedding = self.state_to_embedding(qs)
            sym_emb, vis_emb = jnp.split(embedding, 2, axis=-1)
            so_pred = self.symbolic_decoder(sym_emb)
            vo_pred = self.visual_decoder(vis_emb, training=training)

            r_pred = self.reward_model(qs).squeeze(-1)

            return h_t, (ps, p_mu, p_sigma), (qs, q_mu, q_sigma), (so_pred, vo_pred), r_pred
