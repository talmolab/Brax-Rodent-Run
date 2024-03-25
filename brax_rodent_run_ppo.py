import functools
import jax
from typing import Dict
import wandb

from brax import envs
from brax.training.agents.ppo import train as ppo
from brax.io import model

from Rodent_Env_Brax import Rodent

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import os

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_async_collectives=true '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_highest_priority_async_stream=true '
)
n_gpus = jax.device_count(backend="gpu")
print(f"Using {n_gpus} GPUs")

config = {
    "env_name": "rodent",
    "algo_name": "ppo",
    "task_name": "run",
    "num_envs": 4096*n_gpus,
    "num_timesteps": 500_000_000,
    "eval_every": 1_000_000,
    "episode_length": 1000,
    "batch_size": 4096*n_gpus,
    "learning_rate": 5e-5,
    "terminate_when_unhealthy": True,
    "run_platform": "Harvard",
    "solver": "cg",
    "iterations": 4,
    "ls_iterations": 4,
}

envs.register_environment('rodent', Rodent)

# instantiate the environment
env_name = config["env_name"]
env = envs.get_environment(env_name, 
                           terminate_when_unhealthy=config["terminate_when_unhealthy"],
                           solver=config['solver'],
                           iterations=config['iterations'],
                           ls_iterations=config['ls_iterations'])

# define the jit reset/step functions
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)


train_fn = functools.partial(
    ppo.train, num_timesteps=config["num_timesteps"], num_evals=int(config["num_timesteps"]/config["eval_every"]),
    reward_scaling=1, episode_length=config["episode_length"], normalize_observations=True, action_repeat=5,
    unroll_length=10, num_minibatches=64, num_updates_per_batch=8,
    discounting=0.99, learning_rate=config["learning_rate"], entropy_cost=1e-3, num_envs=config["num_envs"],
    batch_size=config["batch_size"], seed=0
)


run = wandb.init(
    project="vnl_debug",
    config=config,
    notes=f"{config['batch_size']} batchsize, " + 
        f"{config['solver']}, {config['iterations']}/{config['ls_iterations']}"
)


wandb.run.name = f"{config['env_name']}_{config['task_name']}_{config['algo_name']}_{config['run_platform']}"


def wandb_progress(num_steps, metrics):
    metrics["num_steps"] = num_steps
    wandb.log(metrics)
    print(metrics)
    
def policy_params_fn(num_steps, make_policy, params, model_path = './model_checkpoints'):
    os.makedirs(model_path, exist_ok=True)
    model.save_params(f"{model_path}/{num_steps}", params)
    

make_inference_fn, params, _ = train_fn(environment=env, progress_fn=wandb_progress, policy_params_fn=policy_params_fn)


model_path = './model_checkpoints/brax_ppo_rodent_run_finished'
model.save_params(model_path, params)