import os
import numpy as np
import einops
import imageio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import gym
import mujoco_py as mjc
import warnings
import pdb

import raisimpy as raisim
import math
import time
from scipy.spatial.transform import Rotation

from .arrays import to_np
from .video import save_video, save_videos
from ml_logger import logger

from datetime import datetime

from scipy.spatial.transform import Rotation as R

from diffuser.datasets.d4rl import load_environment

#-----------------------------------------------------------------------------#
#------------------------------- helper structs ------------------------------#
#-----------------------------------------------------------------------------#

def env_map(env_name):
    '''
        map D4RL dataset names to custom fully-observed
        variants for rendering
    '''
    if 'halfcheetah' in env_name:
        return 'HalfCheetahFullObs-v2'
    elif 'hopper' in env_name:
        return 'HopperFullObs-v2'
    elif 'walker2d' in env_name:
        return 'Walker2dFullObs-v2'
    else:
        return env_name

#-----------------------------------------------------------------------------#
#------------------------------ helper functions -----------------------------#
#-----------------------------------------------------------------------------#

def get_image_mask(img):
    background = (img == 255).all(axis=-1, keepdims=True)
    mask = ~background.repeat(3, axis=-1)
    return mask

def atmost_2d(x):
    while x.ndim > 2:
        x = x.squeeze(0)
    return x

def quaternion_to_rotation_matrix(quat):
    # 쿼터니언을 회전 행렬로 변환
    q = np.array(quat)
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    rotmat = np.array([
        [1 - 2 * qy ** 2 - 2 * qz ** 2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx ** 2 - 2 * qz ** 2, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx ** 2 - 2 * qy ** 2]
    ])
    return rotmat

def quaternion_from_direction_vector(vec):
    # 벡터의 크기 계산
    norm = np.linalg.norm(vec)

    # 크기가 너무 작으면 단위 쿼터니언 반환
    if norm < 1e-6:
        return np.array([1, 0, 0, 0])

    # 방향 벡터 정규화
    direction = vec / norm

    # Euler 각도 계산
    theta_z = np.arctan2(direction[1], direction[0]) + np.pi / 2  # x축을 회전축에 맞추기
    theta_x = np.arccos(direction[2])  # z축 기울이기

    # Euler 각도를 회전 행렬로 변환
    rot = R.from_euler('zx', [theta_z, theta_x]).as_matrix()

    # 회전 행렬을 쿼터니언으로 변환
    quat = R.from_matrix(rot).as_quat()

    # scipy의 쿼터니언은 (x, y, z, w) 순서이므로 (w, x, y, z) 순서로 변환
    quat = np.array([quat[3], quat[0], quat[1], quat[2]])

    return quat

#-----------------------------------------------------------------------------#
#---------------------------------- renderers --------------------------------#
#-----------------------------------------------------------------------------#

class MuJoCoRenderer:
    '''
        default mujoco renderer
    '''

    def __init__(self, env):
        if type(env) is str:
            env = env_map(env)
            self.env = gym.make(env)
        else:
            self.env = env
        ## - 1 because the envs in renderer are fully-observed
        ## @TODO : clean up
        self.observation_dim = np.prod(self.env.observation_space.shape) - 1
        self.action_dim = np.prod(self.env.action_space.shape)
        try:
            self.viewer = mjc.MjRenderContextOffscreen(self.env.sim)
        except:
            print('[ utils/rendering ] Warning: could not initialize offscreen renderer')
            self.viewer = None

    def pad_observation(self, observation):
        state = np.concatenate([
            np.zeros(1),
            observation,
        ])
        return state

    def pad_observations(self, observations):
        qpos_dim = self.env.sim.data.qpos.size
        ## xpos is hidden
        xvel_dim = qpos_dim - 1
        xvel = observations[:, xvel_dim]
        xpos = np.cumsum(xvel) * self.env.dt
        states = np.concatenate([
            xpos[:,None],
            observations,
        ], axis=-1)
        return states

    def render(self, observation, dim=256, partial=False, qvel=True, render_kwargs=None, conditions=None):

        if type(dim) == int:
            dim = (dim, dim)

        if self.viewer is None:
            return np.zeros((*dim, 3), np.uint8)

        if render_kwargs is None:
            xpos = observation[0] if not partial else 0
            render_kwargs = {
                'trackbodyid': 2,
                'distance': 3,
                'lookat': [xpos, -0.5, 1],
                'elevation': -20
            }

        for key, val in render_kwargs.items():
            if key == 'lookat':
                self.viewer.cam.lookat[:] = val[:]
            else:
                setattr(self.viewer.cam, key, val)

        if partial:
            state = self.pad_observation(observation)
        else:
            state = observation

        qpos_dim = self.env.sim.data.qpos.size
        if not qvel or state.shape[-1] == qpos_dim:
            qvel_dim = self.env.sim.data.qvel.size
            state = np.concatenate([state, np.zeros(qvel_dim)])

        set_state(self.env, state)

        self.viewer.render(*dim)
        data = self.viewer.read_pixels(*dim, depth=False)
        data = data[::-1, :, :]
        return data

    def _renders(self, observations, **kwargs):
        images = []
        for observation in observations:
            img = self.render(observation, **kwargs)
            images.append(img)
        return np.stack(images, axis=0)

    def renders(self, samples, partial=False, **kwargs):
        if partial:
            samples = self.pad_observations(samples)
            partial = False

        sample_images = self._renders(samples, partial=partial, **kwargs)

        composite = np.ones_like(sample_images[0]) * 255

        for img in sample_images:
            mask = get_image_mask(img)
            composite[mask] = img[mask]

        return composite

    def composite(self, savepath, paths, dim=(1024, 256), **kwargs):

        render_kwargs = {
            'trackbodyid': 2,
            'distance': 10,
            'lookat': [5, 2, 0.5],
            'elevation': 0
        }
        images = []
        for path in paths:
            ## [ H x obs_dim ]
            path = atmost_2d(path)
            img = self.renders(to_np(path), dim=dim, partial=True, qvel=True, render_kwargs=render_kwargs, **kwargs)
            images.append(img)
        images = np.concatenate(images, axis=0)

        if savepath is not None:
            fig = plt.figure()
            plt.imshow(images)
            logger.savefig(savepath, fig)
            print(f'Saved {len(paths)} samples to: {savepath}')

        return images

    def render_rollout(self, savepath, states, **video_kwargs):
        if type(states) is list: states = np.array(states)
        images = self._renders(states, partial=True)
        save_video(savepath, images, **video_kwargs)

    def render_plan(self, savepath, actions, observations_pred, state, fps=30):
        ## [ batch_size x horizon x observation_dim ]
        observations_real = rollouts_from_state(self.env, state, actions)

        ## there will be one more state in `observations_real`
        ## than in `observations_pred` because the last action
        ## does not have an associated next_state in the sampled trajectory
        observations_real = observations_real[:,:-1]

        images_pred = np.stack([
            self._renders(obs_pred, partial=True)
            for obs_pred in observations_pred
        ])

        images_real = np.stack([
            self._renders(obs_real, partial=False)
            for obs_real in observations_real
        ])

        ## [ batch_size x horizon x H x W x C ]
        images = np.concatenate([images_pred, images_real], axis=-2)
        save_videos(savepath, *images)

    def render_diffusion(self, savepath, diffusion_path, **video_kwargs):
        '''
            diffusion_path : [ n_diffusion_steps x batch_size x 1 x horizon x joined_dim ]
        '''
        render_kwargs = {
            'trackbodyid': 2,
            'distance': 10,
            'lookat': [10, 2, 0.5],
            'elevation': 0,
        }

        diffusion_path = to_np(diffusion_path)

        n_diffusion_steps, batch_size, _, horizon, joined_dim = diffusion_path.shape

        frames = []
        for t in reversed(range(n_diffusion_steps)):
            print(f'[ utils/renderer ] Diffusion: {t} / {n_diffusion_steps}')

            ## [ batch_size x horizon x observation_dim ]
            states_l = diffusion_path[t].reshape(batch_size, horizon, joined_dim)[:, :, :self.observation_dim]

            frame = []
            for states in states_l:
                img = self.composite(None, states, dim=(1024, 256), partial=True, qvel=True, render_kwargs=render_kwargs)
                frame.append(img)
            frame = np.concatenate(frame, axis=0)

            frames.append(frame)

        save_video(savepath, frames, **video_kwargs)

    def __call__(self, *args, **kwargs):
        return self.renders(*args, **kwargs)

class RaisimRenderer:
    '''
        raisim renderer
    '''

    def __init__(self, env):
        if type(env) is str:
            env = env_map(env)
            self.env = gym.make(env)
        else:
            self.env = env
        ## - 1 because the envs in renderer are fully-observed
        ## @TODO : clean up
        self.observation_dim = np.prod(self.env.observation_space.shape) - 1
        self.action_dim = np.prod(self.env.action_space.shape)

        raisim.World.setLicenseFile("/home/kdyun/.raisim/activation.raisim")
        world_ = raisim.World()
        ground = world_.addGround()

        anymal_ = world_.addArticulatedSystem(os.path.dirname(os.path.abspath(__file__)) + "/../environments/assets/go1/go1.urdf")
        # print(anymal_.getGeneralizedCoordinateDim())

        self.world = world_
        self.anymal = anymal_
        self.dt = 0.02
        self.worldTime = 0


    def pad_observation(self, observation):
        state = np.concatenate([
            np.zeros(1),
            observation,
        ])
        return state

    def pad_observations(self, observations):
        qpos_dim = self.env.sim.data.qpos.size
        ## xpos is hidden
        xvel_dim = qpos_dim - 1

        # go1 dataset version
        xvel_dim = 0
        yvel_dim = 1
        zvel_dim = 2

        xvel = observations[:, xvel_dim]

        # go1 dataset version
        yvel = observations[:, yvel_dim]
        zvel = observations[:, zvel_dim]

        xpos = np.cumsum(xvel) * self.env.dt
        # go1 dataset version
        ypos = np.cumsum(yvel) * self.env.dt
        zpos = np.cumsum(zvel) * self.env.dt

        states = np.concatenate([
            xpos[:,None],
            ypos[:,None], # go1 version
            zpos[:,None], # go1 version
            observations,
        ], axis=-1)
        return states

    def render(self, observation, dim=256, partial=False, qvel=True, render_kwargs=None, conditions=None):

        if type(dim) == int:
            dim = (dim, dim)

        if render_kwargs is None:
            xpos = observation[0] if not partial else 0
            render_kwargs = {
                'trackbodyid': 2,
                'distance': 3,
                'lookat': [xpos, -0.5, 1],
                'elevation': -20
            }

        if partial:
            state = self.pad_observation(observation)
        else:
            state = observation

        qpos_dim = self.env.sim.data.qpos.size
        if not qvel or state.shape[-1] == qpos_dim:
            qvel_dim = self.env.sim.data.qvel.size
            state = np.concatenate([state, np.zeros(qvel_dim)])

        quat = [state[6], state[3], state[4], state[5]]
        siny_cosp = 2 * (quat[0] * quat[3] + quat[1] * quat[2])
        cosy_cosp = 1 - 2 * (quat[2] * quat[2] + quat[3] * quat[3])
        yaw_angle = math.atan2(siny_cosp, cosy_cosp)

        # default_dof_pos = [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5]
        # del_dof_pos = state[7:19] # 7:19
        # dof_pos = default_dof_pos + del_dof_pos

        dof_pos = state[-24:-12]

        # math.cos(yaw_angle)*state[0], math.sin(yaw_angle)*state[0], 0.5,
        # gc = [  math.cos(yaw_angle)*1.5, math.sin(yaw_angle)*1.5, 0.5,
        gc = [  *state[0:3],
                *quat,
                *dof_pos  ]

        # gc = [ state[0], 0.,state[1],
        #        math.cos(state[2]/2), 0., math.sin(state[2]/2), 0.,
        #        state[3], state[4], state[5] ]
        self.world.setWorldTime(self.worldTime)
        self.anymal.setGeneralizedCoordinate(gc)
        time.sleep(0.05)

        data = np.zeros((*dim, 3), np.uint8)
        self.worldTime += self.dt
        return data


    def render_cmd(self, observation, reward, visSphere, visArrow, dim=256, partial=False, qvel=True, render_kwargs=None, conditions=None):

        if type(dim) == int:
            dim = (dim, dim)

        if partial:
            state = self.pad_observation(observation)
        else:
            state = observation

        quat = [state[6], state[3], state[4], state[5]]
        siny_cosp = 2 * (quat[0] * quat[3] + quat[1] * quat[2])
        cosy_cosp = 1 - 2 * (quat[2] * quat[2] + quat[3] * quat[3])
        yaw_angle = math.atan2(siny_cosp, cosy_cosp)

        dof_pos = state[-24:-12]

        # math.cos(yaw_angle)*state[0], math.sin(yaw_angle)*state[0], 0.5,
        # gc = [  math.cos(yaw_angle)*1.5, math.sin(yaw_angle)*1.5, 0.5,
        gc = [  *state[0:3],
                *quat,
                *dof_pos  ]

        q_inv = np.array([state[6], -state[3], -state[4], -state[5]])
        q_inv = q_inv / np.linalg.norm(q_inv)
        Rot_wb = quaternion_to_rotation_matrix(q_inv)
        cmd = np.array([reward[1], reward[2], 0])
        dir_vec = np.dot(Rot_wb, cmd)
        dir_quat = quaternion_from_direction_vector(dir_vec)

        visSphere.setPosition(np.array([state[0], state[1], state[2] + 0.3]))
        visArrow.setPosition(np.array([state[0], state[1], state[2] + 0.2]))
        visArrow.setOrientation(dir_quat)
        visArrow.setCylinderSize(0.2, np.linalg.norm(cmd)/5)

        if reward[0] == -1:
            visSphere.setColor(0, 0, 0, 1)
        elif reward[0] == 0:
            visSphere.setColor(1, 0, 0, 1)
        elif reward[0] == 1:
            visSphere.setColor(1, 1, 0, 1)
        elif reward[0] == 2:
            visSphere.setColor(0, 1, 0, 1)
        elif reward[0] == 3:
            visSphere.setColor(0, 0, 1, 1)

        self.world.setWorldTime(self.worldTime)
        self.anymal.setGeneralizedCoordinate(gc)
        time.sleep(0.05)

        data = np.zeros((*dim, 3), np.uint8)
        self.worldTime += self.dt
        return data

    def _renders(self, observations, **kwargs):
        images = []
        for observation in observations:
            img = self.render(observation, **kwargs)
            images.append(img)
        return np.stack(images, axis=0)

    def _renders_cmd(self, observations, returns, visSphere, visArrow, **kwargs):
        images = []
        for i, observation in enumerate(observations):
            reward = returns[i]
            img = self.render_cmd(observation, reward, visSphere, visArrow, **kwargs)
            images.append(img)
        return np.stack(images, axis=0)

    def renders(self, samples, partial=False, **kwargs):
        if partial:
            samples = self.pad_observations(samples)
            partial = False

        sample_images = self._renders(samples, partial=partial, **kwargs)

        composite = np.ones_like(sample_images[0]) * 255

        for img in sample_images:
            mask = get_image_mask(img)
            composite[mask] = img[mask]

        return composite

    def renders_cmd(self, samples, returns, visSphere, visArrow, partial=False, **kwargs):
        if partial:
            samples = self.pad_observations(samples)
            partial = False

        sample_images = self._renders_cmd(samples, returns, visSphere, visArrow, partial=partial, **kwargs)

        composite = np.ones_like(sample_images[0]) * 255

        for img in sample_images:
            mask = get_image_mask(img)
            composite[mask] = img[mask]

        return composite

    def composite(self, savepath, paths, dim=(1024, 256), **kwargs):

        self.worldTime = 0
        render_kwargs = {
            'trackbodyid': 2,
            'distance': 10,
            'lookat': [5, 2, 0.5],
            'elevation': 0
        }

        server = raisim.RaisimServer(self.world)
        server.launchServer(8088)
        server.focusOn(self.anymal)

        images = []
        for path in paths:
            ## [ H x obs_dim ]
            path = atmost_2d(path)
            img = self.renders(to_np(path), dim=dim, partial=False, qvel=True, render_kwargs=render_kwargs, **kwargs)
            images.append(img)
        images = np.concatenate(images, axis=0)

        if savepath is not None:
            fig = plt.figure()
            plt.imshow(images)
            logger.savefig(savepath, fig)
            print(f'Saved {len(paths)} samples to: {savepath}')

        server.killServer()

        return images

    def composite2(self, savepath, paths, title, dim=(1024, 256), **kwargs):

        render_kwargs = {
            'trackbodyid': 2,
            'distance': 10,
            'lookat': [5, 2, 0.5],
            'elevation': 0
        }

        server = raisim.RaisimServer(self.world)
        server.launchServer(8088)
        server.startRecordingVideo(title+".mp4")
        server.focusOn(self.anymal)
        time.sleep(1)

        images = []
        for path in paths:
            ## [ H x obs_dim ]
            path = atmost_2d(path)
            img = self.renders(to_np(path), dim=dim, partial=False, qvel=True, render_kwargs=render_kwargs, **kwargs)
            images.append(img)
        images = np.concatenate(images, axis=0)

        if savepath is not None:
            fig = plt.figure()
            plt.imshow(images)
            logger.savefig(savepath, fig)
            print(f'Saved {len(paths)} samples to: {savepath}')

        server.stopRecordingVideo()
        server.killServer()

        return images


    def composite3(self, savepath, paths, title, dim=(1024, 256), **kwargs):

        self.worldTime = 0
        render_kwargs = {
            'trackbodyid': 2,
            'distance': 10,
            'lookat': [5, 2, 0.5],
            'elevation': 0
        }

        server = raisim.RaisimServer(self.world)
        server.launchServer(8088)
        current_date = datetime.now().strftime("%m%d")
        directory_path = os.path.expanduser(f'~/workspace/raisim/raisimLib/raisimUnity/linux/Screenshot/{current_date}/{savepath}')
        os.makedirs(directory_path, exist_ok=True)
        video_path = current_date + "/" + savepath + "/" + title + ".mp4"
        server.startRecordingVideo(video_path)
        server.focusOn(self.anymal)
        time.sleep(1)

        images = []
        for path in paths:
            ## [ H x obs_dim ]
            path = atmost_2d(path)
            img = self.renders(to_np(path), dim=dim, partial=False, qvel=True, render_kwargs=render_kwargs, **kwargs)
            images.append(img)
        images = np.concatenate(images, axis=0)

        server.stopRecordingVideo()
        server.killServer()

        return images

    def composite4(self, savepath, paths, rets, title, dim=(1024, 256), **kwargs):

        self.worldTime = 0
        render_kwargs = {
            'trackbodyid': 2,
            'distance': 10,
            'lookat': [5, 2, 0.5],
            'elevation': 0
        }

        server = raisim.RaisimServer(self.world)
        server.launchServer(8088)
        visSphere = server.addVisualSphere("v_sphere", 0.04, 1, 1, 1, 1)
        visArrow = server.addVisualArrow("v_arrow", 0.02, 0.5, 0,0,0,-1, False, False)

        current_date = datetime.now().strftime("%m%d")
        directory_path = os.path.expanduser(f'~/workspace/raisim/raisimLib/raisimUnity/linux/Screenshot/{current_date}/{savepath}')
        os.makedirs(directory_path, exist_ok=True)
        video_path = current_date + "/" + savepath + "/" + title + ".mp4"
        server.focusOn(self.anymal)
        time.sleep(1)
        server.startRecordingVideo(video_path)

        images = []
        for i, path in enumerate(paths):
            ## [ H x obs_dim ]
            path = atmost_2d(path)
            ret = atmost_2d(rets[i])
            img = self.renders_cmd(to_np(path), to_np(ret), visSphere, visArrow, dim=dim, partial=False, qvel=True, render_kwargs=render_kwargs, **kwargs)
            images.append(img)
        images = np.concatenate(images, axis=0)

        server.stopRecordingVideo()
        server.killServer()

        return images


    def composite1(self, savepath, paths, dim=(1024, 256), **kwargs):

        render_kwargs = {
            'trackbodyid': 2,
            'distance': 10,
            'lookat': [5, 2, 0.5],
            'elevation': 0
        }

        server = raisim.RaisimServer(self.world)
        server.launchServer(8080)
        server.focusOn(self.anymal)
        server.startRecordingVideo("eval.mp4")

        images = []
        for path in paths:
            ## [ H x obs_dim ]
            path = atmost_2d(path)
            img = self.renders(to_np(path), dim=dim, partial=True, qvel=True, render_kwargs=render_kwargs, **kwargs)
            images.append(img)
        images = np.concatenate(images, axis=0)

        if savepath is not None:
            fig = plt.figure()
            plt.imshow(images)
            logger.savefig(savepath, fig)
            print(f'Saved {len(paths)} samples to: {savepath}')

        server.stopRecordingVideo()
        server.killServer()

        return images

    def render_rollout(self, savepath, states, **video_kwargs):
        if type(states) is list: states = np.array(states)
        images = self._renders(states, partial=True)
        save_video(savepath, images, **video_kwargs)

    def render_plan(self, savepath, actions, observations_pred, state, fps=30):
        ## [ batch_size x horizon x observation_dim ]
        observations_real = rollouts_from_state(self.env, state, actions)

        ## there will be one more state in `observations_real`
        ## than in `observations_pred` because the last action
        ## does not have an associated next_state in the sampled trajectory
        observations_real = observations_real[:,:-1]

        images_pred = np.stack([
            self._renders(obs_pred, partial=True)
            for obs_pred in observations_pred
        ])

        images_real = np.stack([
            self._renders(obs_real, partial=False)
            for obs_real in observations_real
        ])

        ## [ batch_size x horizon x H x W x C ]
        images = np.concatenate([images_pred, images_real], axis=-2)
        save_videos(savepath, *images)

    def render_diffusion(self, savepath, diffusion_path, **video_kwargs):
        '''
            diffusion_path : [ n_diffusion_steps x batch_size x 1 x horizon x joined_dim ]
        '''
        render_kwargs = {
            'trackbodyid': 2,
            'distance': 10,
            'lookat': [10, 2, 0.5],
            'elevation': 0,
        }

        diffusion_path = to_np(diffusion_path)

        n_diffusion_steps, batch_size, _, horizon, joined_dim = diffusion_path.shape

        frames = []
        for t in reversed(range(n_diffusion_steps)):
            print(f'[ utils/renderer ] Diffusion: {t} / {n_diffusion_steps}')

            ## [ batch_size x horizon x observation_dim ]
            states_l = diffusion_path[t].reshape(batch_size, horizon, joined_dim)[:, :, :self.observation_dim]

            frame = []
            for states in states_l:
                img = self.composite(None, states, dim=(1024, 256), partial=True, qvel=True, render_kwargs=render_kwargs)
                frame.append(img)
            frame = np.concatenate(frame, axis=0)

            frames.append(frame)

        save_video(savepath, frames, **video_kwargs)

    def __call__(self, *args, **kwargs):
        return self.renders(*args, **kwargs)

#-----------------------------------------------------------------------------#
#---------------------------------- rollouts ---------------------------------#
#-----------------------------------------------------------------------------#

def set_state(env, state):
    qpos_dim = env.sim.data.qpos.size
    qvel_dim = env.sim.data.qvel.size
    if not state.size == qpos_dim + qvel_dim:
        warnings.warn(
            f'[ utils/rendering ] Expected state of size {qpos_dim + qvel_dim}, '
            f'but got state of size {state.size}')
        state = state[:qpos_dim + qvel_dim]

    env.set_state(state[:qpos_dim], state[qpos_dim:])

def rollouts_from_state(env, state, actions_l):
    rollouts = np.stack([
        rollout_from_state(env, state, actions)
        for actions in actions_l
    ])
    return rollouts

def rollout_from_state(env, state, actions):
    qpos_dim = env.sim.data.qpos.size
    env.set_state(state[:qpos_dim], state[qpos_dim:])
    observations = [env._get_obs()]
    for act in actions:
        obs, rew, term, _ = env.step(act)
        observations.append(obs)
        if term:
            break
    for i in range(len(observations), len(actions)+1):
        ## if terminated early, pad with zeros
        observations.append( np.zeros(obs.size) )
    return np.stack(observations)
