from typing import Sequence, Tuple
from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen
from jax import numpy as jp
import jax
from brax.training import acting
import numpy as np

from vnl_brax.data import BraxData

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
  vision_network: networks.FeedForwardNetwork # maximize the utilization of ppo
  parametric_action_distribution: distribution.ParametricDistribution


# main function for training
def make_inference_fn(ppo_networks: PPONetworks):
  """Creates params and inference function for the PPO agent."""

  def make_policy(params: types.PolicyParams, deterministic: bool = False) -> types.Policy:
    policy_network = ppo_networks.policy_network
    vision_network = ppo_networks.vision_network
    parametric_action_distribution = ppo_networks.parametric_action_distribution

    # brax obs space passed in from ppo.train.py calling
    def policy(observations: types.Observation, key_sample: PRNGKey) -> Tuple[types.Action, types.Extra]:


       ''' vision processing first'''
       #ToDo: figure out a way to use ppo to train vision_net to step once
       
       vision_raw_obs = observations.vision
       buffer_pro = observations.buffer_proprioception
       vision_buffered = jp.concatenate([vision_raw_obs, buffer_pro], axis=1) # don't need to unmap, can directly concat on axis 1

       print(*params) # tells you the architecture

       # parameter created must have all obs sapce
       vision_param = vision_network.apply(*params, vision_buffered)
       # we actually already have the parameters here, but would it be trained?

       print(vision_param) #should be an 16 value array, (in networks.make_value_network function)


       '''proprioception + vision actiavtions'''
       #ToDo: adding buffer now, but should figure out how to replace later

       proprioception = observations.proprioception
       buffer_vis = observations.buffer_vision
       full_processed = jp.concatenate([proprioception, vision_param,buffer_vis], axis=1) # now type as expected in brax
       logits = policy_network.apply(*params, full_processed)

       print(logits)
       # this is previously a Traced<ShapedArray(float32[16])>with<DynamicJaxprTrace(level=3/0)>
       # now it is a Traced<ShapedArray(float32[128,16])>with<DynamicJaxprTrace(level=3/0)>
       
       #logits = _re_vmap(logits)
       print(observations.shape)

       '''this should be the same with brax implementation from here brax should output a [128, 16] on this level as well'''
       
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
    policy_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    vision_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    activation: networks.ActivationFn = linen.swish) -> PPONetworks:
  
  """Make PPO networks with Vision"""
  #ToDo: Find a way to change the parameter initilization

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
  
  # ToDo: add AlexNet strcuture for vision network change the base_network.py file
  # vision network
  vision_network = networks.make_policy_network(
      param_size=parametric_action_distribution.param_size,
      obs_size=observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=vision_hidden_layer_sizes,
      activation=activation)


  return PPONetworks(
      policy_network=policy_network,
      value_network=value_network,
      vision_network=vision_network,
      parametric_action_distribution=parametric_action_distribution)
