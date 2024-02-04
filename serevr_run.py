# Import
from datetime import datetime
import functools
from IPython.display import HTML
import PIL.Image
import yaml
from typing import List, Dict, Text, Callable, NamedTuple, Optional, Union, Any, Sequence, Tuple
from matplotlib import pyplot as plt
import mediapy as media
import wandb
import os

import numpy as np

from etils import epath
from flax import struct
from ml_collections import config_dict

import mujoco
from mujoco import mjx

from dm_control import mjcf as mjcf_dm
from dm_control import composer
from dm_control.locomotion.examples import basic_rodent_2020
from dm_control.composer.variation import distributions
from dm_control.locomotion.arenas import corridors as corr_arenas
from dm_control.locomotion.tasks import corridors as corr_tasks
from dm_control.locomotion.walkers import rodent, ant
# from dm_control import viewer
from dm_control import mujoco as mujoco_dm

import jax
from jax import numpy as jp
from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.envs.base import Env, MjxEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, model
from brax.io import mjcf as mjcf_brax

# Define inheritence relationships from dm_control
class Gap_Vnl(corr_arenas.GapsCorridor):
    def _build(self, corridor_width, corridor_length, visible_side_planes, aesthetic, platform_length, gap_length):
        super()._build(corridor_width=corridor_width,
                       corridor_length=corridor_length,
                       visible_side_planes=visible_side_planes,
                       aesthetic = aesthetic,
                       platform_length = platform_length,
                       gap_length = gap_length)
    
    def regenerate(self, random_state):
        super().regenerate(random_state)

# Task now just serve as a wrapper
class Task_Vnl(corr_tasks.RunThroughCorridor):
    def __init__(self,
               walker,
               arena,
               walker_spawn_position):
        
        # we don't really need the rest of the reward setup in dm_control, just how the walker is attached to the arena
        spawn_site =  arena._mjcf_root.worldbody.add('site', pos = walker_spawn_position)
        self._arena = arena
        self._walker = walker
        self._walker.create_root_joints(self._arena.attach(self._walker, attach_site=spawn_site)) # customize starting environment

# Define environment class in brax
class Walker(MjxEnv):
  '''
  This is greatly coustomizable of what reward you want to give: reward engineering
  '''
  def __init__(
      self,
      forward_reward_weight=1.25,
      reach_target_reward=5.0, # reward for reaching target area
      ctrl_cost_weight=0.1,
      healthy_reward=5.0,
      terminate_when_unhealthy=True,
      healthy_z_range=(0.2, 1.0), # healthy reward takes care of not falling
      reset_noise_scale=1e-2,
      exclude_current_positions_from_observation=True,
      **kwargs,):
    '''
    Defining initilization of the agent
    '''

    mj_model = physics.model.ptr
    # this is directly a mj_model already of type mujoco_py.MjModel (This is already a MJModel, same as previously in brax)
    # the original xml load is directly creaing an new MjModel instance, which carries the configuration of everything, including mjtCone
    # but this pass in one doesn't, it uses the default mjCONE_PYRAMIDAL, but MjModel now uses the eliptic model, so reset is needed

    # solver is an optimization system
    mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
    mj_model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL # Read documentation

    #Iterations for solver
    mj_model.opt.iterations = 6
    mj_model.opt.ls_iterations = 6

    # Defult framne to be 5, but can self define in kwargs
    physics_steps_per_control_step = 5
    
    kwargs['n_frames'] = kwargs.get(
        'n_frames', physics_steps_per_control_step)

    # Parents inheritence from MjxEnv class
    super().__init__(model=mj_model, **kwargs)

    # Global vraiable for later calling them
    self._forward_reward_weight = forward_reward_weight
    self.reach_target_reward = reach_target_reward
    self._ctrl_cost_weight = ctrl_cost_weight
    self._healthy_reward = healthy_reward
    self._terminate_when_unhealthy = terminate_when_unhealthy
    self._healthy_z_range = healthy_z_range
    self._reset_noise_scale = reset_noise_scale
    self._exclude_current_positions_from_observation = (exclude_current_positions_from_observation)

  def reset(self, rng: jp.ndarray) -> State:
    """Resets the environment to an initial state."""

    #Creating randome keys
    #rng = random number generator key for starting random initiation
    rng, rng1, rng2 = jax.random.split(rng, 3)

    low, hi = -self._reset_noise_scale, self._reset_noise_scale

    #Vectors of generalized joint position in the configuration space
    qpos = self.sys.qpos0 + jax.random.uniform(
        rng1, (self.sys.nq,), minval=low, maxval=hi
    )

    #Vectors of generalized joint velocities in the configuration space
    qvel = jax.random.uniform(
        rng2, (self.sys.nv,), minval=low, maxval=hi
    )

    data = self.pipeline_init(qpos, qvel)

    #Reset everything
    obs = self._get_obs(data.data, jp.zeros(self.sys.nu))
    reward, done, zero = jp.zeros(3)
    metrics = {
        'forward_reward': zero,
        'reward_linvel': zero,
        'reward_quadctrl': zero,
        'reward_alive': zero,
        'x_position': zero,
        'y_position': zero,
        'distance_from_origin': zero,
        'x_velocity': zero,
        'y_velocity': zero,
    }
    return State(data, obs, reward, done, metrics)

  def step(self, state: State, action: jp.ndarray) -> State:
    """Runs one timestep of the environment's dynamics."""
    #Previous Pipeline
    data0 = state.pipeline_state

    #Current pipeline state, step 1
    data = self.pipeline_step(data0, action)

    #Running forward (Velocity)
    com_before = data0.data.subtree_com[1]
    com_after = data.data.subtree_com[1]
    velocity = (com_after - com_before) / self.dt
    forward_reward = self._forward_reward_weight * velocity[0] * 2

    #Reaching the target location
    if jp.linalg.norm(com_after) > jp.linalg.norm(com_before):
      reach_target_reward = 2 * self.reach_target_reward


    #Height being healthy
    min_z, max_z = self._healthy_z_range
    is_healthy = jp.where(data.q[2] < min_z, 0.0, 1.0)
    is_healthy = jp.where(data.q[2] > max_z, 0.0, is_healthy)

    #Termination condition
    if self._terminate_when_unhealthy:
      healthy_reward = self._healthy_reward
    else:
      healthy_reward = self._healthy_reward * is_healthy

    #Control quad cost
    ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

    #Feedback from env
    obs = self._get_obs(data.data, action)
    reward = forward_reward + healthy_reward + reach_target_reward - ctrl_cost

    #Termination State
    done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

    state.metrics.update(
        forward_reward=forward_reward,
        reach_target_reward = reach_target_reward,
        reward_linvel=forward_reward,
        reward_quadctrl=-ctrl_cost,
        reward_alive=healthy_reward,
        x_position=com_after[0],
        y_position=com_after[1],
        distance_from_origin=jp.linalg.norm(com_after),
        x_velocity=velocity[0],
        y_velocity=velocity[1],
    )

    return state.replace(
        pipeline_state=data, obs=obs, reward=reward, done=done
    )

  def _get_obs(self, data: mjx.Data, action: jp.ndarray) -> jp.ndarray:
    """Observes humanoid body position, velocities, and angles."""
    position = data.qpos
    if self._exclude_current_positions_from_observation:
      position = position[2:]

    # external_contact_forces are excluded
    # environment observation described later
    return jp.concatenate([
        position,
        data.qvel,
        data.cinert[1:].ravel(),
        data.cvel[1:].ravel(),
        data.qfrc_actuator,
    ])
  

# Initilizing dm_control
arena = Gap_Vnl(platform_length=distributions.Uniform(.4, .8),
      gap_length=distributions.Uniform(.05, .2),
      corridor_width=5, # walker width follows corridor width
      corridor_length=40,
      aesthetic='outdoor_natural',
      visible_side_planes=False)

arena.regenerate(random_state=None)

walker = ant.Ant(observable_options={'egocentric_camera': dict(enabled=True)})

task = Task_Vnl(
    walker=walker,
    arena=arena,
    walker_spawn_position=(3, 0, 0))

# Export from dm_control
random_state = np.random.RandomState(12345)
task.initialize_episode_mjcf(random_state)
physics = mjcf_dm.Physics.from_mjcf_model(task.root_entity.mjcf_model)

# Brax environment initilization
envs.register_environment('walker', Walker)
env = envs.get_environment(env_name='walker')
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)
state = jit_reset(jax.random.PRNGKey(0))

# Training configuration
config = {
    "env_name": 'walker',
    "algo_name": "ppo",
    "task_name": "gap",
    "num_timesteps": 10_000_000,
    "num_evals": 25,
    "episode_length": 1000,
    "num_envs": 1,
    "batch_size": 512,
    "num_minibatches": 32,
    "num_updates_per_batch": 2,
    "unroll_length": 5,
    }

train_fn = functools.partial(
    ppo.train, num_timesteps=config['num_timesteps'], num_evals=config['num_evals'], reward_scaling=0.1,
    episode_length=config['episode_length'], normalize_observations=True, action_repeat=1,
    unroll_length=config['unroll_length'], num_minibatches=config['num_minibatches'], num_updates_per_batch=config['num_updates_per_batch'],
    discounting=0.97, learning_rate=3e-4, entropy_cost=1e-3, num_envs=config['num_envs'],
    batch_size=config['batch_size'], seed=0)

run = wandb.init(
    project="vnl_task",
    config=config
)

wandb.run.name = f"{config['env_name']}_{config['task_name']}_{config['algo_name']}_brax"


def wandb_progress(num_steps, metrics):
    metrics["num_steps"] = num_steps
    wandb.log(metrics)
    print(metrics)
    
def policy_params_fn(num_steps, make_policy, params, model_path = './model_checkpoints'):
    os.makedirs(model_path, exist_ok=True)
    model.save_params(f"{model_path}/{num_steps}", params)
    
make_inference_fn, params, _ = train_fn(environment=env, progress_fn=wandb_progress, policy_params_fn=policy_params_fn)

model_path = './model_checkpoints/brax_ppo_task_finished'
model.save_params(model_path, params)