import os
import sys

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from absl import app, flags
from ml_collections import config_flags

from sim.neutron_source_one_shot import NeutronSourceOneShot
from trainer.sac_trainer import SACTrainer


FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'NeutronSourceOneShot', 'Environment name.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('num_samples', 72000, 'Number of sampling steps.')
flags.DEFINE_integer('replay_buffer_size', 1000, 'Replay buffer capacity.')
flags.DEFINE_integer('start_training', 100,
                     'Number of training steps to start training.')
flags.DEFINE_integer('episode_length', 1000, 'Max episode length.')
flags.DEFINE_integer('eval_episodes', 1,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 144, 'Eval interval.')
flags.DEFINE_integer('train_interval', 36, 'Mini-batch size for SGD.')
flags.DEFINE_integer('train_batch_size', 32, 'Mini-batch size for SGD.')
flags.DEFINE_integer('train_sgd_iters', 4, 'Mini-batch size for SGD.')
config_flags.DEFINE_config_file(
    'config',
    'configs/sac_default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def main(_):
    config = dict(FLAGS.config)

    create_env = NeutronSourceOneShot
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
                         train_sgd_iters=FLAGS.train_sgd_iters,
                         seed=FLAGS.seed)

    trainer.start()
    trainer.agent.save(trainer.save_dir)
    print(f'Agent saved in dir: {trainer.save_dir}')


if __name__ == '__main__':
    app.run(main)
