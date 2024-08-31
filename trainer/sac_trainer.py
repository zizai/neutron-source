import os
from datetime import datetime
from typing import Dict

import jax
import numpy as np
import ray
import rx
from rx import operators
from tqdm import tqdm

from sac.sac_agent import SACAgent
from trainer.replay_buffer import ReplayBuffer
from trainer.trainer import Trainer


def evaluate(agent, env, num_episodes, **kwargs) -> Dict[str, float]:
    stats = {'total_reward': []}
    for _ in range(num_episodes):
        obs, done = env.reset(), False
        rew = 0
        while not done:
            action = agent.take_action(obs, **kwargs)
            obs, rew, done, info = env.step(action)
            rew += rew

        stats['total_reward'].append(rew)

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats


@ray.remote
class SACWorker(object):
    def __init__(self, create_env, sac_config, seed):
        # initialize environment
        self.env = create_env()
        self.obs = self.env.reset()

        # initialize agent
        self.agent = SACAgent(seed,
                              self.env.observation_space.sample()[np.newaxis],
                              self.env.action_space.sample()[np.newaxis],
                              **sac_config)
        self.worker_timesteps = 0

    def ping(self):
        # import os
        # print("GPU IDs: {}".format(ray.get_runtime_context().get_accelerator_ids()["GPU"]))
        # print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
        return ray.get_runtime_context().get_accelerator_ids()["GPU"], self.worker_timesteps

    def rollout_step(self):
        action = self.agent.sample_action(self.obs)
        next_obs, reward, done, info = self.env.step(action)
        results = self.obs, action, next_obs, reward, done, info

        if done:
            self.obs = self.env.reset()
        else:
            self.obs = next_obs

        self.worker_timesteps += 1
        return results

    def update(self, sac):
        self.agent.sac = sac
        return True


class SACTrainer(Trainer):

    def __init__(self,
                 sac_config,
                 create_env,
                 num_cpus: int = 4,
                 num_samples: int = int(1e7),
                 replay_buffer_size: int = int(1e6),
                 start_training: int = int(1e4),
                 episode_length: int = 1000,
                 eval_interval: int = int(1e5),
                 log_interval: int = int(1e3),
                 train_batch_size: int = 256,
                 train_interval: int = int(1e3),
                 train_sgd_iters: int = 4,
                 save_video: bool = False,
                 seed: int = 42):
        super(SACTrainer, self).__init__(seed, sac_config, save_video=save_video)

        self.num_cpus = num_cpus
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
        self.eval_env = create_env(work_dir='/root/flukawork/neutron_source_' + datetime.now().__format__('%Y-%m-%d_%H-%M-%S'))

        # initialize workers
        num_workers = 36
        ray.init(num_cpus=num_workers, num_gpus=num_workers // 4)
        self.workers = [SACWorker.options(num_cpus=1, num_gpus=0.25).remote(create_env, sac_config, seed * 2 + i) for i in range(num_workers)]

        # initialize agent
        print(jax.devices())
        self.agent = SACAgent(seed,
                              self.eval_env.observation_space.sample()[np.newaxis],
                              self.eval_env.action_space.sample()[np.newaxis],
                              device=jax.devices()[-1],
                              **sac_config)

        self.pbar = None

    def rollout_step(self, t):
        rollout_ids = [worker.rollout_step.remote() for worker in self.workers]
        for _ in range(len(rollout_ids)):
            [rollout_id], rollout_ids = ray.wait(rollout_ids)
            obs, action, next_obs, reward, done, info = ray.get(rollout_id)
            mask = 1. - done
            self.replay_buffer.insert(obs, action, reward, mask, next_obs)

            t += 1
            self.pbar.update(1)
        return t

    def train_step(self, t):
        if t >= self.start_training and t % self.train_interval == 0:
            for j in range(self.train_sgd_iters):
                batch = self.replay_buffer.sample(self.train_batch_size)
                update_info = self.agent.learn_on_batch(batch)

            for worker in self.workers:
                ray.get(worker.update.remote(self.agent.sac))

            for k, v in update_info.items():
                self.summary_writer.add_scalar(f'training/{k}', v, t)
            self.summary_writer.flush()
        return t

    def eval_step(self, t):
        if t >= self.start_training and t % self.eval_interval == 0:
            eval_stats = evaluate(self.agent, self.eval_env, self.eval_episodes)

            for k, v in eval_stats.items():
                self.summary_writer.add_scalar(f'evaluation/{k}', v, t)
            self.summary_writer.flush()
        return t

    def reset(self):
        self.stop()

    def start(self):
        self.pbar = tqdm(total=self.num_samples)
        t = 0
        while t < self.num_samples:
            t = self.rollout_step(t)
            t = self.train_step(t)
            t = self.eval_step(t)

    def stop(self):
        pass
