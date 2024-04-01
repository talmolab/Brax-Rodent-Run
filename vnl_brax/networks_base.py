# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Network definitions."""
# making it a inherent class later！

import dataclasses
from typing import Any, Callable, Sequence, Tuple
import warnings

from brax.training import types
from brax.training.spectral_norm import SNDense
from flax import linen
import jax
import jax.numpy as jnp

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


@dataclasses.dataclass
class FeedForwardNetwork:
  init: Callable[..., Any]
  apply: Callable[..., Any]


class MLP(linen.Module):
  """MLP module, both network (policy and value) have conv_net,
  Maximize flexibility -> creating any model needed"""
  layer_sizes: Sequence[int]
  activation: ActivationFn = linen.relu
  kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
  activate_final: bool = False
  bias: bool = True

  @linen.compact
  def __call__(self, data: jnp.ndarray):
    #print(data.shape) # initial should all be zero

    # two dimension matrix slicing
    vision_data = data[...,data.shape[-1]-12288:] #just vision, ant have 27, rodent have more, change automatically with getting image of 12288
    pro_data = data[...,:data.shape[-1]-12288] #just proprioception

    dtype = jnp.float32
    vision_data = vision_data.astype(dtype) / 255.0
    #print(vision_data.shape)

    # vmap_size = -1 # automatically infered size #vision_data.shape[0]
    # vision_data = vision_data.reshape((vmap_size, 240, 320, 3)) # reshape back to 3d image with vmap considered

    #handling dynamic new shape issues
    #avoid error in case of 1 d as well, add anything that is not the [-1] position, extract all dimensions except the last one
    new_shape_prefix = vision_data.shape[:-1]
    new_shape = new_shape_prefix + (64, 64, 3)
    vision_data = vision_data.reshape(new_shape)
    print(f'Before into ConvNet + vmap shape: {vision_data.shape}')

    vision_data = linen.Conv(features=32,
                      kernel_size=(8, 8),
                      strides=(4, 4),
                      name='conv1',
                      dtype=dtype,
                      )(vision_data)
    vision_data = linen.relu(vision_data)
    vision_data = linen.Conv(features=64,
                      kernel_size=(4, 4),
                      strides=(2, 2),
                      name='conv2',
                      dtype=dtype,
                      )(vision_data)
    vision_data = linen.relu(vision_data)
    vision_data = linen.Conv(features=64,
                      kernel_size=(3, 3),
                      strides=(1, 1),
                      name='conv3',
                      dtype=dtype,
                      )(vision_data)
    vision_data = linen.relu(vision_data)
    
    # flatten preserving expected dimension of 76800, then fit automaticall
    # vision_data = vision_data.reshape((-1, 76800))

    # fully connected layer
    vision_data = linen.Dense(features=512, name='hidden', dtype=dtype)(vision_data)
    vision_data = linen.relu(vision_data)
    vision_out = linen.Dense(features=1, name='logits', dtype=dtype)(vision_data) # this is (2560, 1)
    print(f'This is out of CovNet {vision_out.shape}')

    # handling dynamic new shape issues
    out_new_shape_prefix = pro_data.shape[:-1]
    out_new_shape = out_new_shape_prefix + (-1,)
    vision_out = vision_out.reshape(out_new_shape)

    hidden = jnp.concatenate([pro_data, vision_out], axis=-1)

    for i, hidden_size in enumerate(self.layer_sizes):
      # print(f'hidden_unit_size:{hidden_size}')
      # print(f'hidden_input_size:{hidden.shape}')    
      # hidden size is a integer [hidden_layer_size, parameter size (which is a NormalTanhDistribution)]

      hidden = linen.Dense(
          hidden_size,
          name=f'hidden_{i}',
          kernel_init=self.kernel_init,
          use_bias=self.bias,
          )(hidden)
      if i != len(self.layer_sizes) - 1 or self.activate_final:
        hidden = self.activation(hidden)
      
    print(f'this is out of full ppo network {hidden.shape}')

    return hidden
  


class SNMLP(linen.Module):
  """MLP module with Spectral Normalization."""
  layer_sizes: Sequence[int]
  activation: ActivationFn = linen.relu
  kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
  activate_final: bool = False
  bias: bool = True

  @linen.compact
  def __call__(self, data: jnp.ndarray):
    hidden = data
    for i, hidden_size in enumerate(self.layer_sizes):
      hidden = SNDense(
          hidden_size,
          name=f'hidden_{i}',
          kernel_init=self.kernel_init,
          use_bias=self.bias)(
              hidden)
      if i != len(self.layer_sizes) - 1 or self.activate_final:
        hidden = self.activation(hidden)
    return hidden


def make_policy_network(
    param_size: int,
    obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types
    .identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu) -> FeedForwardNetwork:
  """Creates a policy network."""
  
  policy_module = MLP(
      layer_sizes=list(hidden_layer_sizes) + [param_size], #[27+16], keep the same hidden layer size is 28, param_size is dynamic
      activation=activation,
      kernel_init=jax.nn.initializers.lecun_uniform())

  def apply(processor_params, policy_params, obs):
    obs = preprocess_observations_fn(obs, processor_params)
    return policy_module.apply(policy_params, obs)

  dummy_obs = jnp.zeros((1, obs_size))
  return FeedForwardNetwork(
      init=lambda key: policy_module.init(key, dummy_obs), apply=apply)


def make_value_network(
    obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types
    .identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu) -> FeedForwardNetwork:
  """Creates a policy network."""

  value_module = MLP(
      layer_sizes=list(hidden_layer_sizes) + [1],
      activation=activation,
      kernel_init=jax.nn.initializers.lecun_uniform())

  def apply(processor_params, policy_params, obs):
    obs = preprocess_observations_fn(obs, processor_params)
    return jnp.squeeze(value_module.apply(policy_params, obs), axis=-1)

  dummy_obs = jnp.zeros((1, obs_size))
  return FeedForwardNetwork(
      init=lambda key: value_module.init(key, dummy_obs), apply=apply)


def make_q_network(
    obs_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types
    .identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    n_critics: int = 2) -> FeedForwardNetwork:
  """Creates a value network."""

  class QModule(linen.Module):
    """Q Module."""
    n_critics: int

    @linen.compact
    def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray):
      hidden = jnp.concatenate([obs, actions], axis=-1)
      res = []
      for _ in range(self.n_critics):
        q = MLP(
            layer_sizes=list(hidden_layer_sizes) + [1],
            activation=activation,
            kernel_init=jax.nn.initializers.lecun_uniform())(
                hidden)
        res.append(q)
      return jnp.concatenate(res, axis=-1)

  q_module = QModule(n_critics=n_critics)

  def apply(processor_params, q_params, obs, actions):
    obs = preprocess_observations_fn(obs, processor_params)
    return q_module.apply(q_params, obs, actions)

  dummy_obs = jnp.zeros((1, obs_size))
  dummy_action = jnp.zeros((1, action_size))
  return FeedForwardNetwork(
      init=lambda key: q_module.init(key, dummy_obs, dummy_action), apply=apply)


def make_model(
    layer_sizes: Sequence[int],
    obs_size: int,
    activation: Callable[[jnp.ndarray], jnp.ndarray] = linen.swish,
    spectral_norm: bool = False,
) -> FeedForwardNetwork:
  """Creates a model.

  Args:
    layer_sizes: layers
    obs_size: size of an observation
    activation: activation
    spectral_norm: whether to use a spectral normalization (default: False).

  Returns:
    a model
  """
  warnings.warn(
      'make_model is deprecated, use make_{policy|q|value}_network instead.')
  dummy_obs = jnp.zeros((1, obs_size))
  if spectral_norm:
    module = SNMLP(layer_sizes=layer_sizes, activation=activation)
    model = FeedForwardNetwork(
        init=lambda rng1, rng2: module.init({
            'params': rng1,
            'sing_vec': rng2
        }, dummy_obs),
        apply=module.apply)
  else:
    module = MLP(layer_sizes=layer_sizes, activation=activation)
    model = FeedForwardNetwork(
        init=lambda rng: module.init(rng, dummy_obs), apply=module.apply)
  return model


def make_models(policy_params_size: int,
                obs_size: int) -> Tuple[FeedForwardNetwork, FeedForwardNetwork]:
  """Creates models for policy and value functions.

  Args:
    policy_params_size: number of params that a policy network should generate
    obs_size: size of an observation

  Returns:
    a model for policy and a model for value function
  """
  warnings.warn(
      'make_models is deprecated, use make_{policy|q|value}_network instead.')
  policy_model = make_model([32, 32, 32, 32, policy_params_size], obs_size)
  value_model = make_model([256, 256, 256, 256, 256, 1], obs_size)
  return policy_model, value_model
