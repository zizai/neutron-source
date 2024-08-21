from typing import Sequence, Tuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from neuroblast.datasets import Batch
from neuroblast.models.common import MLP, Params, PRNGKey, default_init, Model, InfoDict


class SymbolicEncoder(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return MLP(self.hidden_dims, activate_final=True)(x)


class SymbolicDecoder(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, z) -> jnp.ndarray:
        return nn.tanh(MLP(self.hidden_dims, activate_final=False)(z))


class Encoder1D(nn.Module):
    training: bool

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        z = x.reshape((x.shape[0], -1, 1))
        z = nn.Conv(features=32, kernel_size=(8,), strides=(2,))(z)
        z = nn.BatchNorm(use_running_average=not self.training, momentum=0.9)(z)
        z = nn.tanh(z)

        z = nn.Conv(features=64, kernel_size=(8,), strides=(2,))(z)
        z = nn.BatchNorm(use_running_average=not self.training, momentum=0.9)(z)
        z = nn.tanh(z)

        z = nn.Conv(features=1, kernel_size=(1,), strides=(1,))(z)
        z = nn.BatchNorm(use_running_average=not self.training, momentum=0.9)(z)
        z = nn.tanh(z)
        z = z.reshape((z.shape[0], -1))
        return z


class Decoder1D(nn.Module):
    training: bool

    @nn.compact
    def __call__(self, z):
        x = z.reshape((z.shape[0], -1, 1))
        x = nn.ConvTranspose(features=64, kernel_size=(8,), strides=(1,))(x)
        x = nn.BatchNorm(use_running_average=not self.training, momentum=0.9)(x)
        x = nn.tanh(x)

        x = nn.ConvTranspose(features=32, kernel_size=(8,), strides=(2,))(x)
        x = nn.BatchNorm(use_running_average=not self.training, momentum=0.9)(x)
        x = nn.tanh(x)

        x = nn.ConvTranspose(features=1, kernel_size=(8,), strides=(2,))(x)
        x = nn.BatchNorm(use_running_average=not self.training, momentum=0.9)(x)
        x = nn.tanh(x)
        x = x.reshape((x.shape[0], -1))
        return x


class VisualEncoder(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        z = nn.Conv(features=32, kernel_size=(4, 4), strides=(2, 2))(x)
        z = nn.BatchNorm(use_running_average=not training, momentum=0.9)(z)
        z = nn.tanh(z)

        z = nn.Conv(features=32, kernel_size=(4, 4), strides=(2, 2))(z)
        z = nn.BatchNorm(use_running_average=not training, momentum=0.9)(z)
        z = nn.tanh(z)

        z = nn.Conv(features=32, kernel_size=(4, 4), strides=(2, 2))(z)
        z = nn.BatchNorm(use_running_average=not training, momentum=0.9)(z)
        z = nn.tanh(z)

        z = nn.Conv(features=1, kernel_size=(4, 4), strides=(2, 2))(z)
        z = nn.BatchNorm(use_running_average=not training, momentum=0.9)(z)
        z = nn.tanh(z)
        return z


class VisualDecoder(nn.Module):
    @nn.compact
    def __call__(self, z: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        B, H = z.shape[0], int(np.sqrt(z.shape[1]))
        assert z.shape[1] % H == 0
        z = z.reshape((B, H, H, 1))
        x = nn.ConvTranspose(features=32, kernel_size=(4, 4), strides=(2, 2))(z)
        x = nn.BatchNorm(use_running_average=not training, momentum=0.9)(x)
        x = nn.tanh(x)

        x = nn.ConvTranspose(features=32, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.BatchNorm(use_running_average=not training, momentum=0.9)(x)
        x = nn.tanh(x)

        x = nn.ConvTranspose(features=32, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.BatchNorm(use_running_average=not training, momentum=0.9)(x)
        x = nn.tanh(x)

        x = nn.ConvTranspose(features=3, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.BatchNorm(use_running_average=not training, momentum=0.9)(x)
        x = nn.tanh(x)
        return x


class WorldModel(nn.Module):
    training: bool = False

    def setup(self):
        self.encoder = VisualEncoder()
        self.decoder = VisualDecoder()

    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        return self.decoder(self.encoder(obs))


@jax.jit
def train_step(model: Model,
               x: jnp.ndarray) -> Tuple[Model, InfoDict]:

    def actor_loss_fn(params, batch_stats) -> Tuple[jnp.ndarray, InfoDict]:
        x_hat, variables = model.apply({'params': params, 'batch_stats': batch_stats}, x, mutable=['batch_stats'])
        loss = jnp.mean((x - x_hat) ** 2)
        return loss, {'loss': loss, 'batch_stats': variables['batch_stats']}

    new_model, info = model.apply_gradient(actor_loss_fn)
    return new_model, info
