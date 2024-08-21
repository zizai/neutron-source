from typing import Tuple

import jax.numpy as jnp
from flax import linen as nn

from models.common import InfoDict, ActorCriticTemp


class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param('log_temp',
                              init_fn=lambda key: jnp.full(
                                  (), jnp.log(self.initial_temperature)))
        return jnp.exp(log_temp)


def update(sac: ActorCriticTemp, entropy: float,
           target_entropy: float) -> Tuple[ActorCriticTemp, InfoDict]:
    def temperature_loss_fn(temp_params):
        temperature = sac.temp.apply({'params': temp_params})
        temp_loss = temperature * (entropy - target_entropy).mean()
        return temp_loss, {'temperature': temperature, 'temp_loss': temp_loss}

    new_temp, info = sac.temp.apply_gradient(temperature_loss_fn)

    new_sac = sac.replace(temp=new_temp)

    return new_sac, info
