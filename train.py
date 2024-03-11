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

import jax
from jax import numpy as jp
from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
# from brax.envs.base import Env, MjxEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
#from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, model
from brax.io import mjcf as mjcf_brax

from vnl_brax.base import Walker
import vnl_brax.networks_vision as ppo_networks_vision
#import vnl_brax.train as ppo


''' Main Training Loop'''

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_async_collectives=true '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_highest_priority_async_stream=true '
)

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
    "num_timesteps": 10_000_00,
    "num_evals": 1000,
    "eval_every": 10_000,
    "episode_length": 1000,
    "num_envs": 32,
    "batch_size": 32,
    "num_minibatches": 32,
    "num_updates_per_batch": 2,
    "unroll_length": 5,
    }

train_fn = functools.partial(
    ppo.train,
    num_timesteps=config['num_timesteps'],
    num_evals=config['num_evals'],
    reward_scaling=0.1,
    episode_length=config['episode_length'],
    normalize_observations=True, action_repeat=1,
    unroll_length=config['unroll_length'],
    num_minibatches=config['num_minibatches'],
    num_updates_per_batch=config['num_updates_per_batch'],
    discounting=0.97, 
    learning_rate=3e-4,
    entropy_cost=1e-3,
    num_envs=config['num_envs'],
    batch_size=config['batch_size'],
    seed=0,
    network_factory=ppo_networks_vision.make_ppo_networks, # This function create the PPO class
    eval_env=None, #we can make an env with normal jax array?
    )

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

model_path = './model_checkpoints/brax_ppo_vision_task_finished'
model.save_params(model_path, params)