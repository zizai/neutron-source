import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


Params = flax.core.FrozenDict[str, Any]
PRNGKey = Any
Shape = Sequence[int]
Dtype = Any  # this could be a real type?
InfoDict = Dict[str, float]


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activate_final: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = nn.relu(x)
        return x


# TODO: Replace with TrainState when it's ready
# https://github.com/google/flax/blob/master/docs/flip/1009-optimizer-api.md#train-state
@flax.struct.dataclass
class Model:
    step: int
    apply_fn: nn.Module = flax.struct.field(pytree_node=False)
    params: Params
    batch_stats: Params
    tx: Optional[optax.GradientTransformation] = flax.struct.field(pytree_node=False)
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(cls,
               model_def: nn.Module,
               inputs: Sequence[jnp.ndarray],
               tx: Optional[optax.GradientTransformation] = None) -> 'Model':
        variables = model_def.init(*inputs)

        params = variables.get('params')
        batch_stats = variables.get('batch_stats')

        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(step=1,
                   apply_fn=model_def,
                   params=params,
                   batch_stats=batch_stats,
                   tx=tx,
                   opt_state=opt_state)

    def __call__(self, *args, **kwargs):
        if self.batch_stats:
            inputs = ({'params': self.params, 'batch_stats': self.batch_stats}, *args)
        else:
            inputs = ({'params': self.params}, *args)
        return self.apply_fn.apply(*inputs, **kwargs)

    def apply(self, *args, **kwargs):
        return self.apply_fn.apply(*args, **kwargs)

    def apply_gradient(self, loss_fn) -> Tuple['Model', InfoDict]:
        grad_fn = jax.grad(loss_fn, has_aux=True)
        if self.batch_stats:
            grads, info = grad_fn(self.params, self.batch_stats)
        else:
            grads, info = grad_fn(self.params)

        if 'batch_stats' in info.keys():
            new_batch_stats = info.pop('batch_stats')
        else:
            new_batch_stats = None

        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(step=self.step + 1,
                            params=new_params,
                            batch_stats=new_batch_stats,
                            opt_state=new_opt_state), info

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load(self, load_path: str) -> 'Model':
        with open(load_path, 'rb') as f:
            params = flax.serialization.from_bytes(self.params, f.read())
        return self.replace(params=params)


@flax.struct.dataclass
class ActorCriticTemp:
    actor: Model
    critic: Model
    target_critic: Model
    temp: Model
    rng: PRNGKey
