# mujoco needed
import mujoco
from mujoco import mjx

# dm_control needed
from dm_control import mjcf as mjcf_dm
from dm_control import composer
from dm_control.composer.variation import distributions
from dm_control.locomotion.arenas import corridors as corr_arenas
from dm_control.locomotion.tasks import corridors as corr_tasks
from dm_control.locomotion.walkers import ant
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
from brax.envs.base import Env, PipelineEnv, State #MjxEnv, State
from brax.mjx.base import State as MjxState
from brax.io import html, model
from brax.io import mjcf as mjcf_brax

# customized import
from vnl_brax.arena import Task_Vnl, Gap_Vnl
import vnl_brax.rodent_base as rodent_base


''' Calling dm_control + Brax Walker Class Adapted'''

arena = Gap_Vnl(platform_length=distributions.Uniform(1.5, 2.0),
      gap_length=distributions.Uniform(.1, .35), # can't be too big, or else can't jump
      corridor_width=10,
      corridor_length=100,
      aesthetic='outdoor_natural',
      visible_side_planes=False)
arena.regenerate(random_state=None)
walker = rodent_base.Rat(observable_options={'egocentric_camera': dict(enabled=True)})

task = Task_Vnl(
    walker=walker,
    arena=arena,
    walker_spawn_position=(1, 0, 0))

# Export from dm_control
random_state = np.random.RandomState(12345)
task.initialize_episode_mjcf(random_state)
physics = mjcf_dm.Physics.from_mjcf_model(task.root_entity.mjcf_model)

# There are quite some big update on Brax_mjx
# MjxEnv is directly an API to the Mujoco mjx
class Walker(PipelineEnv):
  '''
  This is greatly coustomizable of what reward you want to give: reward engineering
  '''
  def __init__(
      self,
      forward_reward_weight=5.0,
      ctrl_cost_weight=0.1,
      healthy_reward=0.5,
      terminate_when_unhealthy=False, # should be false in rendering
      healthy_z_range=(0.0, 1.0), # healthy reward takes care of not falling, this is the contact_termination in dm_control
      train_reward=5.0,
      reset_noise_scale=1e-2,
      exclude_current_positions_from_observation=True,
      solver="cg",
      iterations: int = 6,
      ls_iterations: int = 6,
      vision=False,
      time_since_render: int = 0,
      **kwargs,):
    '''
    Defining initilization of the agent
    '''

    # mj_model = physics.model.ptr
    # this is directly a mj_model already of type mujoco_py.MjModel (This is already a MJModel, same as previously in brax)
    # the original xml load is directly creaing an new MjModel instance, which carries the configuration of everything, including mjtCone
    # but this pass in one doesn't, it uses the default mjCONE_PYRAMIDAL, but MjModel now uses the eliptic model, so reset is needed

    _XML_PATH = "./assets/rodent_optimized.xml"
    mj_model = mujoco.MjModel.from_xml_path(_XML_PATH)

    # solver is an optimization system
    mj_model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL # Read documentation
    mj_model.opt.solver = {
      'cg': mujoco.mjtSolver.mjSOL_CG,
      'newton': mujoco.mjtSolver.mjSOL_NEWTON,
    }[solver.lower()]

    #Iterations for solver
    mj_model.opt.iterations = iterations
    mj_model.opt.ls_iterations = ls_iterations

    sys = mjcf_brax.load_model(mj_model)

    # Defult framne to be 5, but can self define in kwargs
    physics_steps_per_control_step = 3
    
    kwargs['n_frames'] = kwargs.get(
        'n_frames', physics_steps_per_control_step)
    kwargs['backend'] = 'mjx'

    # Parents inheritence from MjxEnv class
    #super().__init__(model=mj_model, **kwargs)
    super().__init__(sys, **kwargs)

    # Global vraiable for later calling them
    self._model = mj_model
    self._forward_reward_weight = forward_reward_weight
    self._ctrl_cost_weight = ctrl_cost_weight
    self._healthy_reward = healthy_reward
    self._terminate_when_unhealthy = terminate_when_unhealthy
    self._healthy_z_range = healthy_z_range
    self._train_reward = train_reward
    self._reset_noise_scale = reset_noise_scale
    self._exclude_current_positions_from_observation = (exclude_current_positions_from_observation)
    self._vision = vision
    self.time_since_render = time_since_render

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
    obs = self._get_obs(data, jp.zeros(self.sys.nu)) #['proprioceptive']
    reward, done, zero = jp.zeros(3)
    metrics = {
        'forward_reward': zero,
        'reward_linvel': zero,
        'reward_quadctrl': zero,
        'reward_alive': zero,
        'x_position': zero,
        'y_position': zero,
        'train_reward': zero,
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
    com_before = data0.subtree_com[3]
    com_after = data.subtree_com[3]

    #print(data.data)
    
    velocity = (com_after - com_before) / self.dt
    forward_reward = self._forward_reward_weight * velocity[0]

    train_reward = self._train_reward * velocity[0] * self.dt # as more training, more rewards

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
    obs = self._get_obs(data, action)
    reward = forward_reward + train_reward + healthy_reward - ctrl_cost

    #print(obs)

    #Termination State
    done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

    state.metrics.update(
        forward_reward=forward_reward,
        reward_linvel=forward_reward,
        reward_quadctrl=-ctrl_cost,
        reward_alive=healthy_reward,
        x_position=com_after[0],
        y_position=com_after[1],
        train_reward=train_reward,
        distance_from_origin=jp.linalg.norm(com_after),
        x_velocity=velocity[0],
        y_velocity=velocity[1],
    )
    return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)

  def _get_obs(self, data: mjx.Data, action: jp.ndarray) -> jp.ndarray:
    """environment feedback of observing walker's proprioreceptive and vision data
    Adapted callback function from Charles (Rodent rendering for wrapping up)"""

    fake_img = np.random.rand(64, 64, 3)
    fake_img =  jax.numpy.array(fake_img).flatten()
    image_jax_noise = fake_img * 1e-12

    if self._vision:
      if (self.time_since_render % 10 == 0):
        def callback(data):
          return self.render(data, height=64, width=64, camera=3)

        img = jax.pure_callback(callback, 
                                np.zeros((64,64,3), dtype=np.uint8), 
                                data)
        img = jax.numpy.array(img).flatten() # 12288 here
        image_jax_noise = img * 1e-12
        #print(f'img shape is {image_jax_noise.shape}')
      
    # Proprioreceptive Data
    position = data.qpos
    velocity = data.qvel
    if self._exclude_current_positions_from_observation:
      position = position[2:]

    proprioception = jp.concatenate([position, velocity])

    self.time_since_render += 1

    return jp.concatenate([proprioception, image_jax_noise])
  
    # if self._vision:
    #     def callback(data):
    #       return self.render(data, height=64, width=64, camera="egocentric")

    #     img = jax.pure_callback(callback, 
    #                             np.zeros((64,64,3), dtype=np.uint8), 
    #                             data)
    #     img = jax.numpy.array(img).flatten()
    #     s = jax.numpy.sum(img) * 1e-12
      
    # else:
    #   s = 0
      
    # # external_contact_forces are excluded
    # return jp.concatenate([
    #     data.qpos + s, data.qvel, 
    #     data.cinert[1:].ravel(),
    #     data.cvel[1:].ravel(),
    #     data.qfrc_actuator
    # ])
