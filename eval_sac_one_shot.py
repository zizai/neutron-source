import os
import sys

from sac.sac_agent import SACAgent
from trainer import sac_trainer

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from absl import app, flags
from ml_collections import config_flags

from sim.neutron_source_one_shot import NeutronSourceOneShot


FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'NeutronSourceOneShot', 'Environment name.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_string('f', None, 'Agent save folder.', required=True)
config_flags.DEFINE_config_file(
    'config',
    'configs/sac_default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def main(_):
    sac_config = dict(FLAGS.config)

    env = NeutronSourceOneShot()
    agent = SACAgent(FLAGS.seed,
                     env.observation_space.sample(),
                     env.action_space.sample(),
                     **sac_config)

    agent.load(FLAGS.f)
    eval_stats = sac_trainer.evaluate(agent, env)
    for k, v in eval_stats.items():
        if isinstance(v, float):
            print(f'{k}: {v}')


if __name__ == '__main__':
    app.run(main)
