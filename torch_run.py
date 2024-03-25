import jax

import torchrl
from torchrl.envs import BraxWrapper

import wandb

from Rodent_Env_Brax import Rodent

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import uuid
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

# Creates the env and an initial starting tensordict
env = BraxWrapper(Rodent, batch_size=[{config['batch_size']}])
td = env.rand_step()


# Generates a completely random UUID (version 4)
run_id = uuid.uuid4()
model_path = f"./model_checkpoints/{run_id}"

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
    
def policy_params_fn(num_steps, make_policy, params, model_path=model_path):
    os.makedirs(model_path, exist_ok=True)
    model.save_params(f"{model_path}/{num_steps}", params)
    

make_inference_fn, params, _ = train_fn(environment=env, progress_fn=wandb_progress, policy_params_fn=policy_params_fn)

final_save_path = f"{model_path}/brax_ppo_rodent_run_finished"
model.save_params(final_save_path, params)
print(f"Run finished. Model saved to {final_save_path}")

# model is from brax.io import model -- what is the torch way to do this