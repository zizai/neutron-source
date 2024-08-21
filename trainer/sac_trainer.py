from typing import Dict

import numpy as np
import rx
from rx import operators
from tqdm import tqdm

from sac.sac_agent import SACAgent
from trainer.replay_buffer import ReplayBuffer
from trainer.trainer import Trainer


def evaluate(agent, env, num_episodes, **kwargs) -> Dict[str, float]:
    stats = {'reward': []}
    for _ in range(num_episodes):
        obs, done = env.reset(), False
        rew = 0
        while not done:
            action = agent.take_action(obs, **kwargs)
            obs, rew, done, info = env.step(action)
            rew += rew

        stats['reward'].append(rew)

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats


class SACTrainer(Trainer):

    def __init__(self,
                 config,
                 create_env,
                 num_samples: int = int(1e7),
                 replay_buffer_size: int = int(1e6),
                 start_training: int = int(1e4),
                 episode_length: int = 1000,
                 eval_interval: int = int(1e5),
                 log_interval: int = int(1e3),
                 train_batch_size: int = 256,
                 train_interval: int = int(1e3),
                 train_sgd_iters: int = 80,
                 save_video: bool = False,
                 seed: int = 42):
        super(SACTrainer, self).__init__(seed, config, save_video=save_video)

        self.num_samples = num_samples
        self.start_training = start_training
        self.episode_length = episode_length
        self.eval_episodes = 1
        self.eval_interval = eval_interval
        self.log_interval = log_interval
        self.train_batch_size = train_batch_size
        self.train_interval = train_interval
        self.train_sgd_iters = train_sgd_iters

        # configure replay buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        # initialize environment
        self.env = create_env()
        self.eval_env = create_env()
        self.observation, self.episode_done = self.env.reset(), False

        # initialize agent
        self.agent = SACAgent(seed,
                              self.env.observation_space.sample().reshape(1, -1),
                              self.env.action_space.sample()[np.newaxis], **config)

        self.rollout_timesteps = 0
        self.pbar = None

    def rollout_step(self, i):
        # if i < self.start_training:
        #     action = self.env.action_space.sample()
        # else:
        #     action = self.agent.take_action(self.observation)
        action = self.agent.take_action(self.observation)

        next_observation, reward, done, info = self.env.step(action)

        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0

        self.replay_buffer.insert(self.observation, action, reward, mask, next_observation)

        self.observation, self.episode_done = next_observation, done
        self.rollout_timesteps += 1
        self.pbar.update(1)

        if self.episode_done:
            self.observation, self.episode_done = self.env.reset(), False
            # for k, v in info['episode'].items():
            #     self.summary_writer.add_scalar(f'training/{k}', v,
            #                                    info['total']['timesteps'])
        return i

    def train_step(self, i):
        if i >= self.start_training and i % self.train_interval == 0:
            for j in range(self.train_sgd_iters):
                batch = self.replay_buffer.sample(self.train_batch_size)
                update_info = self.agent.learn_on_batch(batch)

            for k, v in update_info.items():
                self.summary_writer.add_scalar(f'training/{k}', v, self.rollout_timesteps)
            self.summary_writer.flush()
        return i

    def eval_step(self, i):
        if i % self.eval_interval == 0:
            eval_stats = evaluate(self.agent, self.eval_env, self.eval_episodes, temperature=0.0)

            for k, v in eval_stats.items():
                self.summary_writer.add_scalar(f'evaluation/average_{k}s', v, self.rollout_timesteps)
            self.summary_writer.flush()
        return i

    def reset(self):
        self.stop()
        self.observation, self.episode_done = self.env.reset(), False
        self.rollout_timesteps = 0

    def start(self):
        self.pbar = tqdm(total=self.num_samples)
        rx.range(1, self.num_samples + 1).pipe(
            operators.map(lambda i: self.rollout_step(i)),
            operators.map(lambda i: self.train_step(i)),
            operators.map(lambda i: self.eval_step(i))
        ).subscribe()

    def stop(self):
        pass
