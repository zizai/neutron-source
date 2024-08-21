import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.agent_name = 'planet'

    config.belief_dim = int(200)
    config.hidden_dim = 200
    config.state_dim = 100
    config.embedding_dim = 1024
    config.lr = 3e-4
    config.free_nats = 3.0

    return config
