import enum
import typing

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp

LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0


class MaskType(enum.Enum):
    input = 1
    hidden = 2
    output = 3


@jax.util.cache()
def get_mask(input_dim: int, output_dim: int, randvar_dim: int,
             mask_type: MaskType) -> jnp.DeviceArray:
    """
    Create a mask for MADE.

    See Figure 1 for a better illustration:
    https://arxiv.org/pdf/1502.03509.pdf

    Args:
        input_dim: Dimensionality of the inputs.
        output_dim: Dimensionality of the outputs.
        rand_var_dim: Dimensionality of the random variable.
        mask_type: MaskType.

    Returns:
        A mask.
    """
    if mask_type == MaskType.input:
        in_degrees = jnp.arange(input_dim) % randvar_dim
    else:
        in_degrees = jnp.arange(input_dim) % (randvar_dim - 1)

    if mask_type == MaskType.output:
        out_degrees = jnp.arange(output_dim) % randvar_dim - 1
    else:
        out_degrees = jnp.arange(output_dim) % (randvar_dim - 1)

    in_degrees = jnp.expand_dims(in_degrees, 0)
    out_degrees = jnp.expand_dims(out_degrees, -1)
    return (out_degrees >= in_degrees).astype(jnp.float32).transpose()


class MaskedDense(nn.Dense):
    event_size: int = 1
    mask_type: MaskType = MaskType.hidden
    use_bias: bool = False

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.asarray(inputs, self.dtype)
        kernel = self.param('kernel', self.kernel_init,
                            (inputs.shape[-1], self.features))
        kernel = jnp.asarray(kernel, self.dtype)

        mask = get_mask(*kernel.shape, self.event_size, self.mask_type)
        kernel = kernel * mask

        y = jax.lax.dot_general(inputs,
                                kernel,
                                (((inputs.ndim - 1, ), (0, )), ((), ())),
                                precision=self.precision)
        if self.use_bias:
            bias = self.param('bias', self.bias_init, (self.features, ))
            bias = jnp.asarray(bias, self.dtype)
            y = y + bias
        return y


class MaskedMLP(nn.Module):
    features: typing.Sequence[int]
    activate_final: bool = False

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, conds: jnp.ndarray) -> jnp.ndarray:
        x = inputs
        x_conds = conds
        for i, feat in enumerate(self.features):
            if i == 0:
                mask_type = MaskType.input
            elif i + 1 < len(self.features):
                mask_type = MaskType.hidden
            else:
                mask_type = MaskType.output
            x = MaskedDense(feat,
                            event_size=inputs.shape[-1],
                            mask_type=mask_type)(x)
            x_conds = nn.Dense(feat)(x_conds)
            x = x + x_conds
            if i + 1 < len(self.features) or self.activate_final:
                x = nn.relu(x)
                x_conds = nn.relu(x_conds)
        return x


class Autoregressive(distrax.Distribution):
    def __init__(self, distr_fn: typing.Callable[[jnp.ndarray],
                                                 distrax.Distribution],
                 batch_shape: typing.Tuple[int], event_dim: int):
        self._distr_fn = distr_fn
        self._event_dim = event_dim
        self._batch_shape = batch_shape

    def _sample_n(self, key, n: int) -> jnp.ndarray:
        keys = jax.random.split(key, self._event_dim)

        samples = jnp.zeros((n, *self._batch_shape, self._event_dim),
                            jnp.float32)

        # TODO: Consider rewriting it with nn.scan.
        for i in range(self._event_dim):
            dist = self._distr_fn(samples)
            dim_samples = dist.sample(seed=keys[i])
            samples = jax.ops.index_update(samples, jax.ops.index[..., i],
                                           dim_samples[..., i])

        return samples

    def log_prob(self, values: jnp.ndarray) -> jnp.ndarray:
        return self._distr_fn(values).log_prob(values)

    @property
    def event_shape(self) -> typing.Tuple[int, ...]:
        return (self._event_dim, )


class MADETanhMixturePolicy(nn.Module):
    features: typing.Sequence[int]
    action_dim: int
    num_components: int = 10

    @nn.compact
    def __call__(self,
                 states: jnp.ndarray,
                 temperature: float = 1.0) -> distrax.Distribution:
        is_initializing = not self.has_variable('params', 'means')
        masked_mlp = MaskedMLP(
            (*self.features, 3 * self.num_components * self.action_dim))
        means_init = self.param('means', nn.initializers.normal(1.0),
                                (self.num_components * self.action_dim, ))

        if is_initializing:
            actions = jnp.zeros((*states.shape[:-1], self.action_dim),
                                states.dtype)
            masked_mlp(actions, states)

        def distr_fn(actions: jnp.ndarray) -> distrax.Distribution:
            outputs = masked_mlp(actions, states)
            means, log_scales, logits = jnp.split(outputs, 3, axis=-1)
            means = means + means_init

            log_scales = jnp.clip(log_scales, LOG_STD_MIN, LOG_STD_MAX)

            def reshape(x):
                new_shape = (*x.shape[:-1], self.num_components,
                             actions.shape[-1])
                x = jnp.reshape(x, new_shape)
                return jnp.swapaxes(x, -1, -2)

            means = reshape(means)
            log_scales = reshape(log_scales)
            logits = reshape(logits)

            dist = distrax.Normal(loc=means,
                                  scale=jnp.exp(log_scales) * temperature)

            dist = distrax.MixtureSameFamily(
                distrax.Categorical(logits=logits), dist)

            return distrax.Independent(dist, reinterpreted_batch_ndims=1)

        dist = Autoregressive(distr_fn, states.shape[:-1], self.action_dim)
        return distrax.Transformed(dist, distrax.Block(distrax.Tanh(), 1))
