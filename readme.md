# Brax Tasks - Online Training Script with Proprioreceptor Data Only
***(Adding Vision Currently)***

## Environment Setup
Currently everything is done through pip. TODO for conda env.
1. Clone this repo
2. `pip install -r requirements.txt`
3. `pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
4. `pip install -U numba`

## Training Script
Change the environment setup and hyper-parameter settings in `server_run.py`, currently the config is:

```python
config = {
    "env_name": 'walker',
    "algo_name": "ppo",
    "task_name": "gap",
    "num_timesteps": 10_000_000,
    "num_evals": 1000,
    "eval_every": 10_000,
    "episode_length": 1000,
    "num_envs": 512,
    "batch_size": 512,
    "num_minibatches": 32,
    "num_updates_per_batch": 2,
    "unroll_length": 5,
    }
```

_Caveat:_ On `run.ai` cluster with Nvidia A40, we can only use the `num_envs = 512`.

Use the followings script to run the training.

```bash
python server_run.py
```