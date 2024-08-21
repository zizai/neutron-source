import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.agent_name = 'dreamer'

    config.belief_dim = 512
    config.embedding_dim = 256
    config.hidden_dim = 256
    config.state_dim = 256
    config.horizon = 10

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.dynamics_lr = 3e-4

    config.free_nats = 3.0
    config.gamma = 0.99
    config.target_entropy = None,
    config.tau = 0.005
    config.init_temperature = 1.0

    return config
