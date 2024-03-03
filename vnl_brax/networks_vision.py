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

from vnl_brax.data import BraxData

'''Actor, Value, and Vision Network'''


def _unpmap(v):
  return jax.tree_util.tree_map(lambda x: x[0], v)

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
       
       # ToDo, figure out a way to use ppo to train vision_net to step once
       ''' vision processing first, similar to train.py'''
       vision_raw_obs = observations.vision
       # this mismatch the data class.image (Traced<ShapedArray(float32[128,230400])>with<DynamicJaxprTrace(level=3/0)>), due to vmap
       
       proprioception = observations.proprioception
       vision = observations.vision
       observations_processed = jp.concatenate([_unpmap(proprioception), _unpmap(vision)])

       print(vision_raw_obs)
       print(*params) # tells you the architecture

       vision_param = vision_network.apply(*params, observations_processed)
       # we actually already have the parameters here, but would it be trained?
       # this is a jax.numpy.array of parameter (in networks.make_value_network function)
       
       '''data combined here'''
       proprioception = observations.proprioception
       visions_activation = vision_param

       observations_processed = jp.concatenate([proprioception, visions_activation]) # now type as expected in brax
       
       # same with brax implementation from here
       logits = policy_network.apply(*params, observations_processed)
       
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
  
  """Make PPO networks with preprocessor."""

  parametric_action_distribution = distribution.NormalTanhDistribution(
      event_size=action_size)
  
  # actor network
  policy_network = networks.make_policy_network(
      parametric_action_distribution.param_size,
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
      obs_size=observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=vision_hidden_layer_sizes,
      activation=activation)


  return PPONetworks(
      policy_network=policy_network,
      value_network=value_network,
      vision_network=vision_network,
      parametric_action_distribution=parametric_action_distribution)
