import subprocess 
def slurm_submit(script):
    output = subprocess.check_output("sbatch", input=script, universal_newlines=True)
    job_id = output.strip().split()[-1]
    return job_id

def submit():
    """Submit job to cluster.
    """
    script = f"""#!/bin/bash
#SBATCH -p olveczkygpu,gpu,gpu_requeue,serial_requeue # olveczky,cox,shared,serial_requeue # olveczkygpu,gpu_requeue
#SBATCH --mem=64000 
#SBATCH -c 16
#SBATCH -N 1 
# #SBATCH --constraint="a100"
#SBATCH -t 0-5:00
#SBATCH -J rodenta100
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:4
# # SBATCH -o /slurm/out
# # SBATCH -e /slurm/error
source ~/.bashrc
module load Mambaforge/22.11.1-fasrc01
source activate rl
module load cuda/12.2.0-fasrc01
nvidia-smi
python3 brax_rodent_run_ppo.py
"""
    print(f"Submitting job")
    job_id = slurm_submit(script) 
    print(job_id)

submit()