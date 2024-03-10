# mujoco needed
import mujoco
from mujoco import mjx

# dm_control needed
from dm_control import mjcf as mjcf_dm
from dm_control import composer
from dm_control.locomotion.examples import basic_rodent_2020
from dm_control.composer.variation import distributions
from dm_control.locomotion.arenas import corridors as corr_arenas
from dm_control.locomotion.tasks import corridors as corr_tasks
from dm_control.locomotion.walkers import rodent, ant
# from dm_control import viewer
from dm_control import mujoco as mujoco_dm

# other
import numpy as np
from etils import epath
from flax import struct
from ml_collections import config_dict

# brax needed
import jax
from jax import numpy as jp
from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.envs.base import Env, MjxEnv, State
from brax.mjx.base import State as MjxState
from brax.io import html, model
from brax.io import mjcf as mjcf_brax

# customized import
from vnl_brax.arena import Task_Vnl, Gap_Vnl


''' Calling dm_control + Brax Walker Class Adapted'''


# Initilizing dm_control
arena = Gap_Vnl(platform_length=distributions.Uniform(.4, .8),
      gap_length=distributions.Uniform(.05, .2),
      corridor_width=5, # walker width follows corridor width
      corridor_length=40,
      aesthetic='outdoor_natural',
      visible_side_planes=False)
arena.regenerate(random_state=None)

walker = ant.Ant(observable_options={'egocentric_camera': dict(enabled=True)})

task = Task_Vnl(
    walker=walker,
    arena=arena,
    walker_spawn_position=(3, 0, 0))

# Export from dm_control
random_state = np.random.RandomState(12345)
task.initialize_episode_mjcf(random_state)
physics = mjcf_dm.Physics.from_mjcf_model(task.root_entity.mjcf_model)


# MjxEnv is directly an API to the Mujoco mjx
class Walker(MjxEnv):
  '''
  This is greatly coustomizable of what reward you want to give: reward engineering
  '''
  def __init__(
      self,
      forward_reward_weight=5.0,
      ctrl_cost_weight=0.1,
      healthy_reward=0.5,
      terminate_when_unhealthy=False,
      healthy_z_range=(0.0, 1.0), # healthy reward takes care of not falling, this is the contact_termination in dm_control
      distance_reward=5.0,
      reset_noise_scale=1e-2,
      exclude_current_positions_from_observation=True,
      **kwargs,):
    '''
    Defining initilization of the agent
    '''

    mj_model = physics.model.ptr
    # this is directly a mj_model already of type mujoco_py.MjModel (This is already a MJModel, same as previously in brax)
    # the original xml load is directly creaing an new MjModel instance, which carries the configuration of everything, including mjtCone
    # but this pass in one doesn't, it uses the default mjCONE_PYRAMIDAL, but MjModel now uses the eliptic model, so reset is needed

    # solver is an optimization system
    mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
    mj_model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL # Read documentation

    #Iterations for solver
    mj_model.opt.iterations = 2
    mj_model.opt.ls_iterations = 4

    # Defult framne to be 5, but can self define in kwargs
    physics_steps_per_control_step = 3
    
    kwargs['n_frames'] = kwargs.get(
        'n_frames', physics_steps_per_control_step)

    # Parents inheritence from MjxEnv class
    super().__init__(model=mj_model, **kwargs)

    # Global vraiable for later calling them
    self._model = mj_model
    self._forward_reward_weight = forward_reward_weight
    self._ctrl_cost_weight = ctrl_cost_weight
    self._healthy_reward = healthy_reward
    self._terminate_when_unhealthy = terminate_when_unhealthy
    self._healthy_z_range = healthy_z_range
    self._distance_rewaed = distance_reward
    self._reset_noise_scale = reset_noise_scale
    self._exclude_current_positions_from_observation = (exclude_current_positions_from_observation)

  def reset(self, rng: jp.ndarray) -> State:
    """Resets the environment to an initial state."""

    #Creating randome keys
    #rng = random number generator key for starting random initiation
    rng, rng1, rng2 = jax.random.split(rng, 3)

    low, hi = -self._reset_noise_scale, self._reset_noise_scale

    #Vectors of generalized joint position in the configuration space
    qpos = self.sys.qpos0 + jax.random.uniform(
        rng1, (self.sys.nq,), minval=low, maxval=hi
    )

    #Vectors of generalized joint velocities in the configuration space
    qvel = jax.random.uniform(
        rng2, (self.sys.nv,), minval=low, maxval=hi
    )

    data = self.pipeline_init(qpos, qvel)

    #Reset everything
    obs = self._get_obs(data.data, jp.zeros(self.sys.nu)) #['proprioceptive']
    reward, done, zero = jp.zeros(3)
    metrics = {
        'forward_reward': zero,
        'reward_linvel': zero,
        'reward_quadctrl': zero,
        'reward_alive': zero,
        'x_position': zero,
        'y_position': zero,
        'distance_reward': zero,
        'distance_from_origin': zero,
        'x_velocity': zero,
        'y_velocity': zero,
    }
    return State(data, obs, reward, done, metrics) # State is a big wrapper that contains all information about the environment

  def step(self, state: State, action: jp.ndarray) -> State: # push towards another State
    """Runs one timestep of the environment's dynamics."""
    #Previous Pipeline
    data0 = state.pipeline_state

    #Current pipeline state, step 1
    #Looking at the documenttaion of pipeline_step, "->" means return a modified State
    data = self.pipeline_step(data0, action)

    #Running forward (Velocity) tracking base on center of mass movement
    com_before = data0.data.subtree_com[3]
    com_after = data.data.subtree_com[3]

    #print(data.data)
    
    velocity = (com_after - com_before) / self.dt
    forward_reward = self._forward_reward_weight * velocity[0]

    #Reaching the target location distance
    #distance = state.metrics['distance_from_origin']
    #distance_reward = [self._distance_rewaed * distance if isinstance(distance, int) else 0]
    distance_reward = self._distance_rewaed * velocity[0] * self.dt

    #Height being healthy
    min_z, max_z = self._healthy_z_range
    is_healthy = jp.where(data.q[2] < min_z, 0.0, 1.0)
    is_healthy = jp.where(data.q[2] > max_z, 0.0, is_healthy)

    #Termination condition
    if self._terminate_when_unhealthy:
      healthy_reward = self._healthy_reward
    else:
      healthy_reward = self._healthy_reward * is_healthy

    #Control force cost
    ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

    #Feedback from env
    obs = self._get_obs(data.data, action)
    reward = forward_reward + distance_reward + healthy_reward - ctrl_cost

    print(obs)

    #Termination State
    done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

    state.metrics.update(
        forward_reward=forward_reward,
        reward_linvel=forward_reward,
        reward_quadctrl=-ctrl_cost,
        reward_alive=healthy_reward,
        x_position=com_after[0],
        y_position=com_after[1],
        distance_reward=distance_reward,
        distance_from_origin=jp.linalg.norm(com_after),
        x_velocity=velocity[0],
        y_velocity=velocity[1],
    )
    return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)

  def _get_obs(self, data: mjx.Data, action: jp.ndarray) -> jp.ndarray:
    """environment feedback of observing walker's proprioreceptive and vision data"""

    # Vision Data Mujoco Version
    # passed in data is a pipeline_state.data object, pipeline_state is the sate
    renderer = mujoco.Renderer(model = self._model)

    # this here is the correct format, need qpos in calling
    #d = mjx.get_data(self._model, data)
    d = mujoco.MjData(self._model)

    mujoco.mj_forward(self._model, d)
    renderer.update_scene(d, camera=3) # can call via name too!
    image = renderer.render()
    image_jax = jax.numpy.array(image)
    print(f'image out of mujoco is {image_jax.shape}')
    # cam = mujoco.MjvCamera()

    # fake_image = jax.numpy.array(np.random.rand(64, 64, 3))
    # image_jax = fake_image.flatten() # fit into jp array

    o_height, o_width, _ = image_jax.shape
    c_x,  c_y = o_width//2, o_height//2
    cropped_jax_image = image_jax[c_y-32:c_y+64, c_x-32:c_x+64, :]

    image_jax = cropped_jax_image.flatten()
    image_jax_noise = jax.numpy.sum(image_jax) * 1e-12 # noise added
    print(f'image cropped {image_jax_noise.shape}')

    # Proprioreceptive Data
    position = data.qpos
    velocity = data.qvel
    if self._exclude_current_positions_from_observation:
      position = position[2:]

    proprioception = jp.concatenate([position, velocity])
    
    
    # buffer_proprioception = jax.numpy.array(np.random.rand(27,))

    # num = (230427-(27+16)) # image size - (proprioreception + activation parameter)
    # buffer_vision = jax.numpy.array(np.random.rand(num,))

    # # for shape call in train.py of ppo
    # shape = jp.concatenate([proprioception,image_jax]).shape[0] # shape -1 is one number, give as shape tuple

    # full = jp.concatenate([proprioception,image_jax])
  
    # return BraxData(
    #   proprioception = proprioception,
    #   vision = image_jax,
    #   full=full,
    #   buffer_proprioception = buffer_proprioception,
    #   buffer_vision = buffer_vision,
    #   shape = (128, shape) # this works, but there is a type check in jax
    # )

    return jp.concatenate([proprioception, image_jax_noise])
