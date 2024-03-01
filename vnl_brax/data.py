import mujoco
from mujoco import mjx
import jax
from jax import numpy as jp

'''BraxData Class'''

# This is the direct inherent data class just like mjx.Data
class BraxData(mujoco.mjx._src.dataclasses.PyTreeNode):
    position:jax.Array
    velocity:jax.Array
    image:jax.Array