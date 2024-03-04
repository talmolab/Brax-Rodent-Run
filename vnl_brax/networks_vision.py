from typing import Sequence, Tuple
from brax.training import distribution
#from brax.training import networks
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen
from jax import numpy as jp
import jax
import numpy as np

import vnl_brax.networks_base as networks

'''Actor, Value, and Vision Network'''

# functions to vectorize mapping and un_vectorize mapping
def _unpmap(v):
  return jax.tree_util.tree_map(lambda x: x[0], v)

def _re_vmap(v, new_axis_size=128):
  def replicate_across_new_axis(x):
    return jp.repeat(jp.expand_dims(x, axis=0), new_axis_size, axis=0)
  return jax.tree_util.tree_map(lambda x: jax.vmap(replicate_across_new_axis)(x), v)


# PPO network class data container
@flax.struct.dataclass
class PPONetworks:
  policy_network: networks.FeedForwardNetwork
  value_network: networks.FeedForwardNetwork
  parametric_action_distribution: distribution.ParametricDistribution


# main function for training
def make_inference_fn(ppo_networks: PPONetworks):
  """Creates params and inference function for the PPO agent."""

  def make_policy(params: types.PolicyParams, deterministic: bool = False) -> types.Policy:
    policy_network = ppo_networks.policy_network
    parametric_action_distribution = ppo_networks.parametric_action_distribution

    # brax obs space passed in from ppo.train.py calling
    def policy(observations: types.Observation, key_sample: PRNGKey) -> Tuple[types.Action, types.Extra]:

       logits = policy_network.apply(*params, observations)

       print('logits:',logits) # Traced<ShapedArray(float32[128,16])>with<DynamicJaxprTrace(level=3/0)>

       if deterministic:
         return ppo_networks.parametric_action_distribution.mode(logits), {}
       
       raw_actions = parametric_action_distribution.sample_no_postprocessing(
         logits, key_sample)
       
       log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
       
       postprocessed_actions = parametric_action_distribution.postprocess(
         raw_actions)
       
       return postprocessed_actions, {
          'log_prob': log_prob,
          'raw_action': raw_actions
      }

    return policy

  return make_policy


# return a PPO class that have being instantiated
def make_ppo_networks(
    observation_size: int, # cannot change this, or pipeline broken issue
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types
    .identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    activation: networks.ActivationFn = linen.swish) -> PPONetworks:
  
  """Make PPO networks with Vision"""



  parametric_action_distribution = distribution.NormalTanhDistribution(
      event_size=action_size)
  


  # actor network
  policy_network = networks.make_policy_network(
      param_size=parametric_action_distribution.param_size,
      obs_size=observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=policy_hidden_layer_sizes,
      activation=activation)
  


  # critic network
  value_network = networks.make_value_network(
      obs_size=observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=value_hidden_layer_sizes,
      activation=activation)
  


  return PPONetworks(
      policy_network=policy_network,
      value_network=value_network,
      parametric_action_distribution=parametric_action_distribution)
