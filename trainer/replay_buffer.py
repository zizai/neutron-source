from typing import Optional, Union

import numpy as np

from trainer.dataset import Dataset, Batch


class ReplayBuffer(object):
    def __init__(self, capacity: int):
        self.size = 0
        self.insert_index = 0
        self.capacity = capacity

        self.observations = [None] * self.capacity
        self.actions = [None] * self.capacity
        self.rewards = [None] * self.capacity
        self.masks = [None] * self.capacity
        self.next_observations = [None] * self.capacity
        self.beliefs = [None] * self.capacity
        self.states = [None] * self.capacity

    def initialize_with_dataset(self, dataset: Dataset,
                                num_samples: Optional[int]):
        assert self.insert_index == 0, 'Can insert a batch online in an empty replay buffer.'

        dataset_size = len(dataset.observations)

        if num_samples is None:
            num_samples = dataset_size
        else:
            num_samples = min(dataset_size, num_samples)
        assert self.capacity >= num_samples, 'Dataset cannot be larger than the replay buffer capacity.'

        if num_samples < dataset_size:
            perm = np.random.permutation(dataset_size)
            indices = perm[:num_samples]
        else:
            indices = np.arange(num_samples)

        self.observations = [r for r in dataset.observations[indices]]
        self.actions = [r for r in dataset.actions[indices]]
        self.rewards = [r for r in dataset.rewards[indices]]
        self.masks = [r for r in dataset.masks[indices]]
        self.next_observations = [r for r in dataset.next_observations[indices]]

        if dataset.beliefs:
            self.beliefs = [r for r in dataset.beliefs[indices]]
        if dataset.states:
            self.states = [r for r in dataset.states[indices]]

        self.insert_index = num_samples
        self.size = num_samples

    def insert(self,
               observation: np.ndarray,
               action: np.ndarray,
               reward: float,
               mask: float,
               next_observation: np.ndarray,
               belief: Optional[np.ndarray] = None,
               state: Optional[np.ndarray] = None):
        self.observations[self.insert_index] = observation
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.masks[self.insert_index] = mask
        self.next_observations[self.insert_index] = next_observation

        if belief is not None:
            self.beliefs[self.insert_index] = belief
        if state is not None:
            self.states[self.insert_index] = state

        self.insert_index = (self.insert_index + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def replay(self, seq_len):
        start = np.random.randint(0, self.size - seq_len)
        idx = range(start, start + seq_len)
        return self.__getitem__(idx)

    def sample(self, batch_size: int) -> Batch:
        idx = np.random.randint(self.size, size=batch_size)
        return self.__getitem__(idx)

    def __getitem__(self, idx):
        o = []
        a = []
        r = []
        masks = []
        next_o = []
        h = []
        s = []
        for i in idx:
            o.append(self.observations[i])
            a.append(self.actions[i])
            r.append(self.rewards[i])
            masks.append(self.masks[i])
            next_o.append(self.next_observations[i])

            if self.beliefs[i] is not None:
                h.append(self.beliefs[i])
            if self.states[i] is not None:
                s.append(self.states[i])

        o, next_o = np.asarray(o), np.asarray(next_o)

        return Batch(observations=o,
                     actions=np.asarray(a),
                     rewards=np.asarray(r),
                     masks=np.asarray(masks),
                     next_observations=next_o,
                     beliefs=np.asarray(h) if h else None,
                     states=np.asarray(s) if s else None)
