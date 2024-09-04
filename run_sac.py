import os
import sys
from datetime import datetime
from functools import partial

from absl import app, flags
from ml_collections import config_flags

from sim.neutron_source_env import NeutronSourceEnv
from trainer.sac_trainer import SACTrainer


FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'NeutronSourceEnv', 'Environment name.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_string('task', 'run', 'Training task name.')
flags.DEFINE_integer('num_samples', int(1e6), 'Number of sampling steps.')
flags.DEFINE_integer('replay_buffer_size', int(1e5), 'Replay buffer capacity.')
flags.DEFINE_integer('start_training', 10000,
                     'Number of training steps to start training.')
flags.DEFINE_integer('episode_length', 1000, 'Max episode length.')
flags.DEFINE_integer('eval_episodes', 1,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 1080, 'Eval interval.')
flags.DEFINE_integer('train_interval', 360, 'Mini-batch size for SGD.')
flags.DEFINE_integer('train_batch_size', 512, 'Mini-batch size for SGD.')
flags.DEFINE_boolean('save_video', True, 'Save videos during evaluation.')
flags.DEFINE_string('gpu_id', '0', 'GPU ID.')
config_flags.DEFINE_config_file(
    'config',
    'configs/sac_default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def main(_):
    config = dict(FLAGS.config)

    create_env = NeutronSourceEnv
    trainer = SACTrainer(config,
                         create_env,
                         num_samples=FLAGS.num_samples,
                         replay_buffer_size=FLAGS.replay_buffer_size,
                         start_training=FLAGS.start_training,
                         episode_length=FLAGS.episode_length,
                         eval_interval=FLAGS.eval_interval,
                         log_interval=FLAGS.log_interval,
                         train_interval=FLAGS.train_interval,
                         train_batch_size=FLAGS.train_batch_size,
                         save_video=FLAGS.save_video,
                         seed=FLAGS.seed)

    trainer.start()


if __name__ == '__main__':
    import os
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    app.run(main)
