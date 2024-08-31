import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.agent_name = 'sac'
    config.log_dir = '/tf_logs'

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4

    config.hidden_dims = (512, 512, 512)

    config.discount = 0.99

    config.tau = 0.005
    config.target_update_period = 1

    config.init_temperature = 1.0
    config.target_entropy = None

    return config
