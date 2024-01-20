#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[1]:


import numpy as np

from datetime import datetime
import functools
from IPython.display import HTML
import jax
from jax import numpy as jp
import numpy as np
from typing import Any, Dict, Sequence, Tuple, Union
import wandb

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.envs.base import Env, MjxEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model

from etils import epath
from flax import struct
from matplotlib import pyplot as plt
import mediapy as media
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import os

import yaml
from typing import List, Dict, Text


# ## Load the model

# ## TODO:
# 
# - Check the healthy z-range of the rodent. Now the training
#     - Check mj_data and how to pull out kinematics of the simulations
# - Check the `brax.envs` and how I can pass the custom parameters

# In[3]:


def load_params(param_path: Text) -> Dict:
    with open(param_path, "rb") as file:
        params = yaml.safe_load(file)
    return params


params = load_params("params/params.yaml")

class Rodent(MjxEnv):
    
    # Might want to change the terminate_when_unhealthy params to enables
    # longer episode length, since the average episode length is too short (1 timestep)
    # temp change the `terminate_when_unhealthy` to extend the episode length.
    def __init__(
            self,
            forward_reward_weight=5,
            ctrl_cost_weight=0.1,
            healthy_reward=0.5,
            terminate_when_unhealthy=False,
            healthy_z_range=(0.2, 1.0),
            reset_noise_scale=1e-2,
            exclude_current_positions_from_observation=False,
            **kwargs,
    ):
        params = load_params("params/params.yaml")
        mj_model = mujoco.MjModel.from_xml_path(params["XML_PATH"])
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6

        physics_steps_per_control_step = 5
        kwargs['n_frames'] = kwargs.get(
            'n_frames', physics_steps_per_control_step)

        super().__init__(model=mj_model, **kwargs)

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.qpos0 + jax.random.uniform(
            rng1, (self.sys.nq,), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(
            rng2, (self.sys.nv,), minval=low, maxval=hi
        )

        data = self.pipeline_init(qpos, qvel)

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
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)
        # based on the timestep simulation, calculate the rewards
        com_before = data0.data.subtree_com[1]
        com_after = data.data.subtree_com[1]
        velocity = (com_after - com_before) / self.dt
        forward_reward = self._forward_reward_weight * velocity[0]

        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(data.q[2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(data.q[2] > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy

        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

        obs = self._get_obs(data.data, action)
        reward = forward_reward + healthy_reward - ctrl_cost
        # terminates when unhealthy
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        state.metrics.update(
            forward_reward=forward_reward,
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

    def _get_obs(
            self, data: mjx.Data, action: jp.ndarray
    ) -> jp.ndarray:
        """Observes humanoid body position, velocities, and angles."""
        position = data.qpos
        if self._exclude_current_positions_from_observation:
            position = position[2:]
            
        # external_contact_forces are excluded
        return jp.concatenate([
            position,
            data.qvel,
            data.cinert[1:].ravel(),
            data.cvel[1:].ravel(),
            data.qfrc_actuator,
        ])


# ## training loop

# In[5]:


envs.register_environment('rodent', Rodent)

# instantiate the environment
env_name = 'rodent'
env = envs.get_environment(env_name)

# define the jit reset/step functions
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

# Change config to conservative measure for debug purposes.
# change eval func to make test the checkpoints
config = {
    "env_name": env_name,
    "algo_name": "ppo",
    "task_name": "run",
    "num_envs": 128,
    "num_timesteps": 10_000,
    "eval_every": 10,
    "episode_length": 500,
    "num_evals": 1000,
    "batch_size": 64,
    "learning_rate": 6e-4,
    "terminate_when_unhealthy": False,
    "run_platform": "runai",
}


train_fn = functools.partial(
    ppo.train, num_timesteps=config["num_timesteps"], num_evals=int(config["num_timesteps"]/config["eval_every"]),
    reward_scaling=0.1, episode_length=config["episode_length"], normalize_observations=True, action_repeat=1,
    unroll_length=10, num_minibatches=8, num_updates_per_batch=4,
    discounting=0.98, learning_rate=config["learning_rate"], entropy_cost=1e-3, num_envs=config["num_envs"],
    batch_size=config["batch_size"], seed=0)

run = wandb.init(
    project="vnl",
    config=config
)


wandb.run.name = f"{config['env_name']}_{config['task_name']}_{config['algo_name']}_{config['run_platform']}_brax"


def wandb_progress(num_steps, metrics):
    metrics["num_steps"] = num_steps
    wandb.log(metrics)
    print(metrics)
    
def policy_params_fn(num_steps, make_policy, params, model_path = './model_checkpoints/brax_ppo_rodent_run'):
    os.makedirs("./model_checkpoints")
    model.save_params(f"{model_path}/{num_steps}", params)
    

make_inference_fn, params, _= train_fn(environment=env, progress_fn=wandb_progress, policy_params_fn=policy_params_fn)


#@title Save Model
model_path = './model_checkpoints/brax_ppo_rodent_run_finished'
model.save_params(model_path, params)