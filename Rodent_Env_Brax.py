import jax
from jax import numpy as jp

from brax.envs.base import PipelineEnv, State
from brax.io import mjcf as mjcf_brax

from dm_control import mjcf as mjcf_dm

import mujoco
from mujoco import mjx

import numpy as np

import os

_XML_PATH = "./models/rodent_new.xml"


class Rodent(PipelineEnv):

    def __init__(
        self,
        track_pos: jp.ndarray,
        forward_reward_weight=10,
        ctrl_cost_weight=0.1,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.03, 0.5),
        reset_noise_scale=1e-2,
        solver="cg",
        iterations: int = 6,
        ls_iterations: int = 6,
        vision=False,
        **kwargs,
    ):
        # Load the rodent model via dm_control
        # dm_rodent = rodent.Rodent()
        # physics = mjcf_dm.Physics.from_mjcf_model(dm_rodent.mjcf_model)
        # mj_model = physics.model.ptr
        os.environ["MUJOCO_GL"] = "osmesa"
        mj_model = mujoco.MjModel.from_xml_path(_XML_PATH)
        mj_model.opt.solver = {
            "cg": mujoco.mjtSolver.mjSOL_CG,
            "newton": mujoco.mjtSolver.mjSOL_NEWTON,
        }[solver.lower()]
        mj_model.opt.iterations = iterations
        mj_model.opt.ls_iterations = ls_iterations

        mj_model.opt.jacobian = 0

        sys = mjcf_brax.load_model(mj_model)

        physics_steps_per_control_step = (
            10  # 10 times 0.002 = 0.02 => fps of tracking data
        )

        kwargs["n_frames"] = kwargs.get("n_frames", physics_steps_per_control_step)
        kwargs["backend"] = "mjx"

        super().__init__(sys, **kwargs)

        self_track_pos = track_pos
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._vision = vision

    def reset(self, rng) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2, rng_pos = jax.random.split(rng, 4)

        start_frame = jax.random.randint(rng, (), 0, 100)

        info = {
            "cur_frame": start_frame,
        }

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = jp.array(self.sys.qpos0).at[:3].set(
            self._track_pos[start_frame]
        ) + jax.random.uniform(rng1, (self.sys.nq,), minval=low, maxval=hi)
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)

        data = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(data, jp.zeros(self.sys.nu))
        reward, done, zero = jp.zeros(3)
        metrics = {
            "pos_reward": zero,
            "reward_quadctrl": zero,
            "reward_alive": zero,
        }
        return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        info = state.info.copy()
        info["cur_frame"] += 1

        pos_reward = jp.exp(
            -100
            * (
                jp.linalg.norm(
                    data.qpos[:3] - (self._track_pos[state.info["cur_frame"]])
                )
            )
        )

        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(data.q[2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(data.q[2] > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy

        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

        obs = self._get_obs(data, action, info["cur_frame"])
        reward = pos_reward + healthy_reward - ctrl_cost
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        state.metrics.update(
            pos_reward=pos_reward,
            reward_quadctrl=-ctrl_cost,
            reward_alive=healthy_reward,
        )

        return state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done, info=info
        )

    def _get_obs(
        self, data: mjx.Data, action: jp.ndarray, cur_frame: int
    ) -> jp.ndarray:
        """Observes rodent body position, velocities, and angles."""
        # get relative tracking position in local frame
        track_pos_local = self.emil_to_local(
            data, self._track_pos[cur_frame + 1] - data.qpos[:3]
        )
        track_pos_local = jp.concatenate([o.flatten() for o in track_pos_local])

        # external_contact_forces are excluded
        return jp.concatenate(
            [
                data.qpos,
                data.qvel,
                data.cinert[1:].ravel(),
                data.cvel[1:].ravel(),
                data.qfrc_actuator,
                track_pos_local,
            ]
        )

    def emil_to_local(self, data, vec_in_world_frame):
        xmat = jp.reshape(data.xmat[1], (3, 3))
        return xmat @ vec_in_world_frame

    def to_local(self, data, vec_in_world_frame):
        """Linearly transforms a world-frame vector into entity's local frame.

        Note that this function does not perform an affine transformation of the
        vector. In other words, the input vector is assumed to be specified with
        respect to the same origin as this entity's local frame. This function
        can also be applied to matrices whose innermost dimensions are either 2 or
        3. In this case, a matrix with the same leading dimensions is returned
        where the innermost vectors are replaced by their values computed in the
        local frame.

        Returns the resulting vector, converting to ego-centric frame
        """
        # TODO: confirm index
        xmat = jp.reshape(data.xmat[1], (3, 3))
        # The ordering of the np.dot is such that the transformation holds for any
        # matrix whose final dimensions are (2,) or (3,).

        # Each element in xmat is a 3x3 matrix that describes the rotation of a body relative to the global coordinate frame, so
        # use rotation matrix to dot the vectors in the world frame, transform basis
        if vec_in_world_frame.shape[-1] == 2:
            return jp.dot(vec_in_world_frame, xmat[:2, :2])
        elif vec_in_world_frame.shape[-1] == 3:
            return jp.dot(vec_in_world_frame, xmat)
        else:
            raise ValueError(
                "`vec_in_world_frame` should have shape with final "
                "dimension 2 or 3: got {}".format(vec_in_world_frame.shape)
            )
