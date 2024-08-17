import numpy as np


class NeutronSource(object):
    def __init__(self, action_dim=64, max_steps=128):
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.geometry = np.ones((max_steps, action_dim))
        self.steps = 0

    def step(self, action):
        self.geometry[self.steps] = action
        self.steps += 1

        rew = self.get_reward()
        done = self.steps == self.max_steps

        return self.geometry, rew, done

    def get_reward(self):
        if self.steps == self.max_steps:
            return 1
        else:
            return 0.

    def reset(self):
        self.geometry = np.ones((self.max_steps, self.action_dim))
        self.steps = 0
        return self.geometry
