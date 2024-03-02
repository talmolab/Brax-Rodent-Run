import mujoco
from mujoco import mjx
import jax
from jax import numpy as jp

'''BraxData Class'''

# This is the direct inherent data class just like mjx.Data
# Essentially we are plugging in a mid step to store all the data and then in the network function give ppo's original data concatinated form

class BraxData(mujoco.mjx._src.dataclasses.PyTreeNode):
    position:jax.Array
    velocity:jax.Array
    image:jax.Array

    def __init__(self, size, position, velocity, image):
        self.size = size
        self.position = position
        self.velocity = velocity
        self.image = image
    
    # for the shape function in calling
    def shape(self):
        return (self.size,)