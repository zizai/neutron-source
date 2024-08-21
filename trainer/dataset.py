import abc
import collections
from typing import Optional

import numpy as np

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations', 'beliefs', 'states'])


class Dataset(object):
    def __init__(self, size: int, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray, next_observations: np.ndarray,
                 beliefs: Optional[np.ndarray] = None, states: Optional[np.ndarray] = None):
        self.size = size
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.next_observations = next_observations
        self.beliefs = beliefs
        self.states = states

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=self.next_observations[indx],
                     beliefs=self.beliefs[indx] if self.beliefs else None,
                     states=self.states[indx] if self.states else None)
