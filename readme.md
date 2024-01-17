# Brax Rodent Run - Online Training Script

## Environment Setup

Currently everything is done through pip. TODO for conda env.

1. Clone this repo

1. `pip install -r requirements.txt`
1. `pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
1. `pip install -U numba`

## Training Script

Change the environment setup and hyper-parameter settings in `brax_rodent_run_ppo.py`, currently the config is:

```python
config = {
    "env_name": env_name,
    "algo_name": "ppo",
    "task_name": "run",
    "num_envs": 2048,
    "num_timesteps": 10_000_000,
    "eval_every": 10_000,
    "episode_length": 1000,
    "num_evals": 1000,
    "batch_size": 512,
    "learning_rate": 3e-4,
    "terminate_when_unhealthy": False
}
```

_Caveat:_ On `run.ai` cluster with Nvidia A40, we can only use the `num_envs = 512`.

Use the followings script to run the training.

```bash
python brax_rodent_run.ppo
```