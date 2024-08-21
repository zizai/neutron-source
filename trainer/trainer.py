import os
import random
from datetime import datetime
from typing import Optional, Dict

import numpy as np
from tensorboardX import SummaryWriter

from utils.logger import setup_logger


class Trainer(object):
    def __init__(self,
                 seed: int,
                 config: Dict,
                 save_video: bool = False,
                 gpu_mem_fraction: Optional[float] = None):
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        if gpu_mem_fraction is not None:
            assert 0 < gpu_mem_fraction < 1
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(gpu_mem_fraction)

        self.agent_name = config.pop('agent_name')
        log_dir = config.pop('log_dir') or os.path.expanduser('~/nb_results')
        self.save_dir = os.path.join(log_dir,
                                     '{}_{}'.format(self.agent_name, datetime.now().__format__('%Y-%m-%d_%H-%M-%S')))

        self.logger = setup_logger()
        self.summary_writer = SummaryWriter(os.path.join(self.save_dir, 'tb'))

        if save_video:
            self.video_folder = os.path.join(self.save_dir, 'video', 'eval')
        else:
            self.video_folder = None

        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def train_step(self, *args, **kwargs):
        raise NotImplementedError

    def eval_step(self, *args, **kwargs):
        raise NotImplementedError

    def rollout_step(self, *args, **kwargs):
        raise NotImplementedError

    def start(self):
        raise NotImplementedError
