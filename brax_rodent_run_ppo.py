import functools
import jax
from typing import Dict
import wandb
import imageio
import mujoco
from brax import envs
from brax.training.agents.ppo import train as ppo
from brax.io import model
import numpy as np
from Rodent_Env_Brax import Rodent
import pickle
import warnings
from preprocessing.mjx_preprocess import process_clip_to_train
from jax import numpy as jp

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
from absl import app
from absl import flags

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"

FLAGS = flags.FLAGS

n_gpus = jax.device_count(backend="gpu")
print(f"Using {n_gpus} GPUs")

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true " "--xla_gpu_triton_gemm_any=True "
)

flags.DEFINE_enum("solver", "cg", ["cg", "newton"], "constraint solver")
flags.DEFINE_integer("iterations", 4, "number of solver iterations")
flags.DEFINE_integer("ls_iterations", 4, "number of linesearch iterations")
flags.DEFINE_boolean("vision", False, "render vision in obs")

config = {
    "env_name": "rodent",
    "algo_name": "ppo",
    "task_name": "run",
    "num_envs": 1024 * n_gpus,
    "num_timesteps": 500_000_000,
    "eval_every": 5_000_000,
    "episode_length": 150,
    "batch_size": 1024 * n_gpus,
    "learning_rate": 5e-5,
    "terminate_when_unhealthy": True,
    "run_platform": "Harvard",
    "solver": "cg",
    "iterations": 6,
    "ls_iterations": 6,
    "vision": False,
}

envs.register_environment("rodent", Rodent)

reference_path = f"clips/84.p"

if os.path.exists(reference_path):
    with open(reference_path, "rb") as file:
        # Use pickle.load() to load the data from the file
        reference_clip = pickle.load(file)
else:
    # Process rodent clip and save as pickle
    reference_clip = process_clip_to_train(
        stac_path="../stac-mjx/transform_snips_new.p",
        start_step=84 * 250,
        clip_length=250,
        mjcf_path="./models/rodent_new.xml",
    )
    with open(reference_path, "wb") as file:
        # Use pickle.dump() to save the data to the file
        pickle.dump(reference_clip, file)

# instantiate the environment
env_name = config["env_name"]
env = envs.get_environment(
    env_name,
    track_pos=reference_clip.position,
    terminate_when_unhealthy=config["terminate_when_unhealthy"],
    solver=config["solver"],
    iterations=config["iterations"],
    ls_iterations=config["ls_iterations"],
    vision=config["vision"],
)

# define the jit reset/step functions
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)


train_fn = functools.partial(
    ppo.train,
    num_timesteps=config["num_timesteps"],
    num_evals=int(config["num_timesteps"] / config["eval_every"]),
    reward_scaling=1,
    episode_length=config["episode_length"],
    normalize_observations=True,
    action_repeat=1,
    unroll_length=10,
    num_minibatches=64,
    num_updates_per_batch=8,
    discounting=0.97,
    learning_rate=config["learning_rate"],
    entropy_cost=1e-3,
    num_envs=config["num_envs"],
    batch_size=config["batch_size"],
    seed=0,
)

import uuid

# Generates a completely random UUID (version 4)
run_id = uuid.uuid4()
model_path = f"./model_checkpoints/{run_id}"

run = wandb.init(project="vnl_debug", config=config, notes="")


wandb.run.name = (
    f"{config['env_name']}_{config['task_name']}_{config['algo_name']}_{run_id}"
)


def wandb_progress(num_steps, metrics):
    metrics["num_steps"] = num_steps
    wandb.log(metrics, commit=True)


def policy_params_fn(num_steps, make_policy, params, model_path=model_path):
    policy_params_key = jax.random.PRNGKey(0)
    os.makedirs(model_path, exist_ok=True)
    model.save_params(f"{model_path}/{num_steps}", params)
    jit_inference_fn = jax.jit(make_policy(params, deterministic=True))
    _, policy_params_key = jax.random.split(policy_params_key)
    reset_rng, act_rng = jax.random.split(policy_params_key)

    state = jit_reset(reset_rng)

    rollout = [state.pipeline_state]
    for i in range(500):
        _, act_rng = jax.random.split(act_rng)
        obs = state.obs
        ctrl, extras = jit_inference_fn(obs, act_rng)
        state = jit_step(state, ctrl)
        rollout.append(state.pipeline_state)

    # Render the walker with the reference expert demonstration trajectory
    os.environ["MUJOCO_GL"] = "osmesa"
    qposes_rollout = [data.qpos for data in rollout]

    def f(x):
        if len(x.shape) != 1:
            return jax.lax.dynamic_slice_in_dim(
                x,
                0,
                250,
            )
        return jp.array([])

    ref_traj = jax.tree_util.tree_map(f, reference_clip)
    qposes_ref = jp.hstack([ref_traj.position, ref_traj.quaternion, ref_traj.joints])

    mj_model = mujoco.MjModel.from_xml_path(f"./models/rodent_pair.xml")

    mj_model.opt.solver = {
        "cg": mujoco.mjtSolver.mjSOL_CG,
        "newton": mujoco.mjtSolver.mjSOL_NEWTON,
    }["cg"]
    mj_model.opt.iterations = 6
    mj_model.opt.ls_iterations = 6
    mj_data = mujoco.MjData(mj_model)

    # save rendering and log to wandb
    os.environ["MUJOCO_GL"] = "osmesa"
    mujoco.mj_kinematics(mj_model, mj_data)
    renderer = mujoco.Renderer(mj_model, height=512, width=512)

    frames = []
    # render while stepping using mujoco
    video_path = f"{model_path}/{num_steps}.mp4"

    with imageio.get_writer(video_path, fps=float(1.0 / env.dt)) as video:
        for qpos1, qpos2 in zip(qposes_ref, qposes_rollout):
            mj_data.qpos = np.append(qpos1, qpos2)
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(mj_data, camera=f"close_profile")
            pixels = renderer.render()
            video.append_data(pixels)
            frames.append(pixels)

    wandb.log({"eval/rollout": wandb.Video(video_path, format="mp4")})


make_inference_fn, params, _ = train_fn(
    environment=env, progress_fn=wandb_progress, policy_params_fn=policy_params_fn
)

final_save_path = f"{model_path}/brax_ppo_rodent_run_finished"
model.save_params(final_save_path, params)
print(f"Run finished. Model saved to {final_save_path}")
