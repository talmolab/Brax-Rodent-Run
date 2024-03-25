import torchrl
from torchrl.envs import BraxWrapper

from Rodent_Env_Brax import Rodent

# Creates the env and an initial starting tensordict
env = Rodent(env_name="rodent")
td = env.rand_step()
