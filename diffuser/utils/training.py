import isaacgym
assert isaacgym

import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
import einops
import pdb
import diffuser
from copy import deepcopy

from .arrays import batch_to_device, to_np, to_device, apply_dict, to_torch
from .timer import Timer
from .cloud import sync_logs
from ml_logger import logger
from ..models.helpers import apply_conditioning

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from adamp import AdamP
import random

import glob
import pickle as pkl
from go1_gym.envs import *
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv

def load_play_env(headless=False):
    dirs = glob.glob(f"../runs/gait-conditioned-agility/pretrain-v0/train/*")
    logdir = sorted(dirs)[0]

    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        print(cfg.keys())

        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(Cfg, key), key2, value2)

    # turn off DR for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    Cfg.terrain.num_rows = 5
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True

    Cfg.domain_rand.lag_timesteps = 1
    Cfg.domain_rand.randomize_lag_timesteps = False
    Cfg.control.control_type = "actuator_net"

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=Cfg)

    return env

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        eval_freq=1000,
        record_freq=50000,
        label_freq=100000,
        save_parallel=False,
        n_reference=4,
        bucket=None,
        train_device='cuda',
        save_checkpoints=False,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.save_checkpoints = save_checkpoints

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.record_freq = record_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset

        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer
        # self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)
        self.optimizer = torch.optim.AdamW(diffusion_model.parameters(), lr=train_lr, weight_decay=1e-3)
        # self.optimizer = AdamP(diffusion_model.parameters(), lr=train_lr, betas=(0.9, 0.999), weight_decay=1e-2)

        self.bucket = bucket
        self.n_reference = n_reference

        self.reset_parameters()
        self.step = 0

        self.device = train_device
        # self.env = load_play_env(headless=False)
        self.env = None
        self.action_scale = dataset.action_scale

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())
        # self.load()

        # state_dict = torch.load(loadpath, map_location=Config.device)
        # self.ema_model.load_state_dict(state_dict['model'])

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):
        state_scale = (self.dataset.normalizer.unnormalize(np.ones(self.dataset.observation_dim), 'observations')
                       - self.dataset.normalizer.unnormalize(-np.ones(self.dataset.observation_dim), 'observations'))
        writer = SummaryWriter()
        timer = Timer()
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch, device=self.device)
                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()

                if step % 2000 == 0:
                    diff_losses, inv_loss, x_targ, x_pred, a_targ, a_pred = self.model.loss2(*batch)
                    diff_loss = diff_losses.mean()

                    normed_loss = (to_np(x_targ) - to_np(x_pred)) ** 2
                    normed_loss = np.mean(normed_loss, axis=(0,1))
                    targ_unnormed = self.dataset.normalizer.unnormalize(to_np(x_targ), 'observations')
                    pred_unnormed = self.dataset.normalizer.unnormalize(to_np(x_pred), 'observations')
                    org_loss = (targ_unnormed - pred_unnormed) ** 2
                    org_loss = np.mean(org_loss, axis=(0,1))

                    a_unnormed = to_np(a_targ) * self.action_scale
                    a_pred_unnormed = to_np(a_pred) * self.action_scale
                    unnormed_inv_loss = (a_pred_unnormed - a_unnormed) ** 2
                    unnormed_inv_loss = np.mean(unnormed_inv_loss, axis=(0,1))

                    writer.add_scalar("loss/diff loss", diff_loss, step)
                    writer.add_scalar("loss/inv loss", inv_loss, step)
                    writer.add_scalar("loss/unnormed inv loss", unnormed_inv_loss, step)

                    writer.add_scalar("error/base_pos[m]", np.sqrt(np.sum(normed_loss[0:2])), step)
                    writer.add_scalar("error/base_ori[quat]", np.sqrt(np.sum(org_loss[3:7])), step)
                    writer.add_scalar("error/base_lin_vel[m/s]", np.sqrt(np.sum(org_loss[7:10])), step)
                    writer.add_scalar("error/base_ang_vel[rad/s]", np.sqrt(np.sum(org_loss[10:13])), step)
                    writer.add_scalar("error/joint_pos[rad]", np.sqrt(np.sum(org_loss[-24:-12])), step)
                    writer.add_scalar("error/joint_vel[rad/s]", np.sqrt(np.sum(org_loss[-12:])), step)

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                self.save()

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                logger.print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}')
                metrics = {k:v.detach().item() for k, v in infos.items()}
                metrics['steps'] = self.step
                metrics['loss'] = loss.detach().item()
                logger.log_metrics_summary(metrics, default_stats='mean')

            if self.step == 0 and self.sample_freq:
                self.render_reference(self.n_reference)

            if self.step and self.step % self.record_freq == 0:
                self.record_samples()

            self.step += 1
        writer.close()

    def save(self):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.bucket, logger.prefix, 'checkpoint')
        os.makedirs(savepath, exist_ok=True)
        # logger.save_torch(data, savepath)
        if self.save_checkpoints:
            savepath = os.path.join(savepath, f'state_{self.step}.pt')
        else:
            savepath = os.path.join(savepath, 'state.pt')
        torch.save(data, savepath)
        logger.print(f'[ utils/training ] Saved model to {savepath}')

    def load(self):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.bucket, logger.prefix, f'checkpoint/state.pt')
        # data = logger.load_torch(loadpath)
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    def evaluate(self, use_warmstarting):
        device = self.device

        observation_dim = self.dataset.observation_dim
        action_dim = self.dataset.action_dim

        gaits = {"pronking": [0, 0, 0],
                 "trotting": [0.5, 0, 0],
                 "bounding": [0, 0.5, 0],
                 "pacing": [0, 0, 0.5]}

        env = self.env

        done = 0
        sampling_time = 0

        #######################  gait version  #############################
        gait_idx = 1
        random_gaits = list(gaits.values())[gait_idx]
        gait = torch.tensor(random_gaits)
        step_frequency = 3.0

        returns = to_device(torch.Tensor([[gait_idx, 1.5, 0, 0]]), device)
        #####################################################################

        default_pos = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5])

        t = 0
        test_step = 150 if use_warmstarting else 50
        ################ play env obdim 42 ver #######################
        obs_list = []
        self.env.commands[:, 4] = step_frequency
        self.env.commands[:, 5:8] = gait
        obs_list.append(self.env.reset().detach().cpu())
        init_xy = self.env.root_states[:,0:2]

        obs = np.concatenate(obs_list, axis=0)
        obs = np.concatenate([to_np(self.env.root_states[:, 0:2]) - to_np(init_xy), to_np(self.env.root_states[:, 2:3]), to_np(self.env.root_states[:, 3:7]),
                              to_np(self.env.root_states[:, 7:10]), to_np(self.env.base_ang_vel[:, :]),
                              obs[:, 66:70],
                              obs[:, 18:30] + default_pos, obs[:, 30:42]], axis=-1)
        ##############################################################

        recorded_obs = [deepcopy(obs[:, None])]
        warm_sample = None
        warm = use_warmstarting

        while t < test_step:
            obs = self.dataset.normalizer.normalize(obs, 'observations')
            conditions = {0: to_torch(obs, device=device)}

            if not warm:
                samples = self.ema_model.conditional_sample(conditions, returns)

            elif warm:
                if t == 0:
                    with torch.no_grad():
                        samples = self.ema_model.conditional_sample(conditions, returns)
                        warm_sample = samples
                ############################# mixing gait ##############################################
                else:
                    if t < 500:
                        with torch.no_grad():
                            samples = self.ema_model.conditional_warm_sample(conditions, warm_sample, k=1, steps=10,
                                                                                returns=returns)
                        warm_sample = samples
                ########################################################################################
            obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)
            obs_comb = obs_comb.reshape(-1, 2 * observation_dim)
            action = self.ema_model.inv_model(obs_comb)

            samples = to_np(samples)
            action = to_np(action)

            action = self.dataset.normalizer.unnormalize(action, 'actions')
            action = to_torch(action[None])

            obs_list = []
            with torch.no_grad():
                self.env.commands[:, 4] = step_frequency
                self.env.commands[:, 5:8] = gait
                this_obs, this_reward, this_done, _ = env.step(action[0])
            this_obs = this_obs.detach().cpu()

            ######################### play env obdim 42 ver ##################################
            this_obs = np.concatenate(
                [to_np(self.env.root_states[:, 0:2]) - to_np(init_xy), to_np(self.env.root_states[:, 2:3]), to_np(self.env.root_states[:, 3:7]),
                 to_np(self.env.root_states[:, 7:10]), to_np(self.env.base_ang_vel[:, :]),
                 this_obs[:, 66:70],
                 this_obs[:, 18:30] + default_pos, this_obs[:, 30:42]], axis=-1)
            ##################################################################################
            obs_list.append(this_obs)
            if this_done:
                if done == 1:
                    pass
                else:
                    done = 1
            else:
                if done == 1:
                    pass

            obs = np.concatenate(obs_list, axis=0)
            recorded_obs.append(deepcopy(obs[:, None]))
            t += 1
        print("end of evaluation")
        return t
    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:,None]

        ## [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.dataset.action_dim:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')
        scaled_xy = normed_observations[:, :, 0:2]
        observations = np.concatenate([scaled_xy, observations[:, :, 2:]], axis=-1)

        # from diffusion.datasets.preprocessing import blocks_cumsum_quat
        # # observations = conditions + blocks_cumsum_quat(deltas)
        # observations = conditions + deltas.cumsum(axis=1)

        #### @TODO: remove block-stacking specific stuff
        # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
        # observations = blocks_add_kuka(observations)
        ####

        savepath = os.path.join('images', f'sample-reference.png')
        self.renderer.composite(savepath, observations)

    def render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, self.device)
            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            if self.ema_model.returns_condition:
                returns = to_device(torch.ones(n_samples, 1), self.device)
            else:
                returns = None

            if self.ema_model.model.calc_energy:
                samples = self.ema_model.grad_conditional_sample(conditions, returns=returns)
            else:
                samples = self.ema_model.conditional_sample(conditions, returns=returns)

            samples = to_np(samples)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = samples[:, :, self.dataset.action_dim:]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]

            # from diffusion.datasets.preprocessing import blocks_cumsum_quat
            # observations = conditions + blocks_cumsum_quat(deltas)
            # observations = conditions + deltas.cumsum(axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            #### @TODO: remove block-stacking specific stuff
            # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
            # observations = blocks_add_kuka(observations)
            ####

            savepath = os.path.join('images', f'sample-{i}.png')
            self.renderer.composite(savepath, observations)

    def inv_render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, self.device)
            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            if self.ema_model.returns_condition:
                # returns = to_device( 0.9 * torch.ones(n_samples, 1), self.device)
                ############################ change the gait here ######################################
                returns = to_device(torch.Tensor([[1, 1.5, 0, 0]
                                                  for i in range(n_samples)]), self.device)
                #########################################################################################
            else:
                returns = None

            if self.ema_model.model.calc_energy:
                samples = self.ema_model.grad_conditional_sample(conditions, returns=returns)
            else:
                samples = self.ema_model.conditional_sample(conditions, returns=returns)

            samples = to_np(samples)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = samples[:, :, :]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]

            # from diffusion.datasets.preprocessing import blocks_cumsum_quat
            # observations = conditions + blocks_cumsum_quat(deltas)
            # observations = conditions + deltas.cumsum(axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            #### @TODO: remove block-stacking specific stuff
            # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
            # observations = blocks_add_kuka(observations)
            ####

            savepath = os.path.join('images', f'sample-{i}.png')
            self.renderer.composite(savepath, observations)

    def record_samples(self, batch_size=2, n_samples=4):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, self.device)
            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            commands = [[1, 1.5, 0, 0], [2, 1.5, 0, 0], [3, 1.5, 0, 0], [0, 1.5, 0, 0]]
            # commands = [[1, 0.8, 0, 0], [1, -0.8, 0, 0], [1, 0, 0.4, 0], [1, 0, 0, 0.8]]
            # commands = [[2, 0.8, 0, 0], [2, -0.8, 0, 0], [2, 0, 0.4, 0], [2, 0, 0, 0.8]]
            # commands = [[3, 0.8, 0, 0], [3, -0.8, 0, 0], [3, 0, 0.4, 0], [3, 0, 0, 0.8]]

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            if self.ema_model.returns_condition:
                # returns = to_device( 0.9 * torch.ones(n_samples, 1), self.device)
                ############################ change the gait here ######################################
                returns = to_device(torch.Tensor([commands[i]
                                                  for i in range(n_samples)]), self.device)
                #########################################################################################
            else:
                returns = None

            samples = self.ema_model.conditional_sample(conditions, returns=returns)
            samples = to_np(samples)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = samples[:, :, :]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]

            # from diffusion.datasets.preprocessing import blocks_cumsum_quat
            # observations = conditions + blocks_cumsum_quat(deltas)
            # observations = conditions + deltas.cumsum(axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')
            scaled_xy = normed_observations[:, :, 0:2]
            observations = np.concatenate([scaled_xy, observations[:, :, 2:]], axis=-1)

            savepath = 'train'
            name = str(self.step)
            self.renderer.composite3(savepath, observations, name)

    def record_samples_acc(self, batch_size=1, n_samples=4):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, self.device)
            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            commands = [[1, 1.5, 0, 0], [2, 1.5, 0, 0], [3, 1.5, 0, 0], [0, 1.5, 0, 0]]
            # commands = [[1, 0.8, 0, 0], [1, -0.8, 0, 0], [1, 0, 0.4, 0], [1, 0, 0, 0.8]]
            # commands = [[2, 0.8, 0, 0], [2, -0.8, 0, 0], [2, 0, 0.4, 0], [2, 0, 0, 0.8]]
            # commands = [[3, 0.8, 0, 0], [3, -0.8, 0, 0], [3, 0, 0.4, 0], [3, 0, 0, 0.8]]

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            if self.ema_model.returns_condition:
                # returns = to_device( 0.9 * torch.ones(n_samples, 1), self.device)
                ############################ change the gait here ######################################
                returns = to_device(torch.Tensor([commands[i]
                                                  for i in range(n_samples)]), self.device)
                #########################################################################################
            else:
                returns = None

            samples = self.ema_model.conditional_sample_acc(conditions, returns=returns)
            samples = to_np(samples)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = samples[:, :, :]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]

            # from diffusion.datasets.preprocessing import blocks_cumsum_quat
            # observations = conditions + blocks_cumsum_quat(deltas)
            # observations = conditions + deltas.cumsum(axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')
            scaled_xy = normed_observations[:, :, 0:2]
            observations = np.concatenate([scaled_xy, observations[:, :, 2:]], axis=-1)

            savepath = 'train'
            name = str(self.step)
            self.renderer.composite3(savepath, observations, name)

class Discriminator(nn.Module):
    def __init__(self, state_dim=56*37, command_dim=4):
        super().__init__()
        input_dim = state_dim + command_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x, commands):
        x_flat = x.reshape(x.size(0), -1)
        combined = torch.cat([x_flat, commands], dim=-1)
        logits = self.model(combined)
        return 2 * torch.sigmoid(logits) - 1

class ADDTrainer(object):
    def __init__(
            self,
            student,
            teacher,
            dataset,
            renderer,
            ema_decay=0.995,
            train_batch_size=32,
            train_lr=2e-5,
            gradient_accumulate_every=2,
            step_start_ema=2000,
            update_ema_every=10,
            log_freq=100,
            sample_freq=1000,
            save_freq=1000,
            eval_freq=1000,
            record_freq=50000,
            label_freq=100000,
            save_parallel=False,
            n_reference=4,
            bucket=None,
            train_device='cuda',
            save_checkpoints=False,
    ):
        super().__init__()
        self.student = student
        self.teacher = teacher

        self.discriminator = Discriminator().to(train_device)

        self.student_timesteps = [25,50,75,99]

        params = [p for name, p in self.student.named_parameters() if not name.startswith("inv_model")]
        self.optimizer = torch.optim.AdamW(params, lr=train_lr, weight_decay=1e-3)
        self.d_optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=train_lr, weight_decay=1e-3)

        for param in teacher.parameters():
            param.requires_grad = False
        for name, p in student.named_parameters():
            if name.startswith("inv_model"):
                p.requires_gard = False
        for name, param in student.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}")

        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.student)
        self.update_ema_every = update_ema_every
        self.save_checkpoints = save_checkpoints

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.record_freq = record_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset

        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer

        self.bucket = bucket
        self.n_reference = n_reference

        self.reset_parameters()
        self.step = 0

        self.device = train_device
        self.env = None
        self.action_scale = dataset.action_scale

    def adversarial_loss_student(self, x_theta, commands):
        """Compute adversarial loss for the student model."""
        logits = self.discriminator(x_theta, commands)
        loss = -torch.mean(logits)
        return loss

    def adversarial_loss_discriminator(self, x_real, x_fake, commands):
        """Compute adversarial loss for the discriminator."""
        real_loss = torch.mean(torch.relu(1.0 - self.discriminator(x_real, commands)))
        fake_loss = torch.mean(torch.relu(1.0 + self.discriminator(x_fake, commands)))
        return 0.5 * (real_loss + fake_loss)

    def reset_parameters(self):
        # self.ema_model.load_state_dict(self.student.state_dict())
        self.load()

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.student)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):
        writer = SummaryWriter()
        timer = Timer()
        for step in range(n_train_steps):
            batch = next(self.dataloader)
            batch = batch_to_device(batch, device=self.device)

            # original state trajectory
            x, cond, command = batch
            x = x[:,:,12:]
            batch_size = len(x)

            interval_mapping = {
                25: (1, 25),
                50: (25, 50),
                75: (50, 75),
                99: (75, 99)
            }



            # noisy state sampling for student
            s = torch.full((batch_size,), random.choice(self.student_timesteps), device=x.device).long()
            noise1 = torch.randn_like(x)
            xs = self.student.q_sample(x_start=x, t=s, noise=noise1)
            xs = apply_conditioning(xs, cond, 0)
            x_theta = self.student.model(xs, cond, s, command)
            x_theta = apply_conditioning(x_theta, cond, 0)
            # x_theta = self.student.p_sample_(xs, cond, s, command)

            # if step == 1:
            #     samples = to_np(x_theta)
            #     normed_observations = samples[0:1, :, :]
            #     observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')
            #     scaled_xy = normed_observations[:, :, 0:2]
            #     observations = np.concatenate([scaled_xy, observations[:, :, 2:]], axis=-1)
            #     savepath = 'train'
            #     name = str('x_theta')
            #     self.renderer.composite3(savepath, observations, name)

            # L_adv = self.adversarial_loss_student(x_theta, command)

            start, end = interval_mapping[s[0].item()]
            # t = torch.randint(0, 100, (batch_size,), device=self.device).long()
            rand_t = random.choice(range(start+1, end+1))
            t = torch.full((batch_size,), rand_t, device=self.device).long()
            noise2 = torch.randn_like(x_theta)
            xt = self.teacher.q_sample(x_start=x_theta, t=t, noise=noise2)
            xt = apply_conditioning(xt, cond, 0)
            with torch.no_grad():
                # x_psi = self.teacher.p_sample_loop_slice(xt, cond, rand_t, command)
                x_psi = self.teacher.model(xt, cond, t, command)
                x_psi = apply_conditioning(x_psi, cond, 0)


            # # reconstruction quality test
            # if step == 0:
            #     for diff_t in [1,33,66,99]:
            #         s = torch.full((batch_size,), diff_t, device=x.device).long()
            #         noise1 = torch.randn_like(x)
            #         xs = self.student.q_sample(x_start=x, t=s, noise=noise1)
            #         xs = apply_conditioning(xs, cond, 0)
            #         x_theta = self.student.model(xs, cond, s, command)
            #         x_theta = apply_conditioning(x_theta, cond, 0)
            #         print(diff_t, " 1 step: " , nn.MSELoss()(x_theta, x).item())
            #
            #         if diff_t == 99:
            #             s = torch.full((batch_size,), 1, device=x.device).long()
            #             noisy_x_theta = self.student.q_sample(x_start=x_theta, t=s, noise=noise1)
            #             noisy_x_theta = apply_conditioning(noisy_x_theta, cond, 0)
            #             pred = self.student.model(noisy_x_theta, cond, s, command)
            #             pred = apply_conditioning(pred, cond, 0)
            #             print("2 step: " , nn.MSELoss()(pred, x).item())
            #
            #             s = torch.full((batch_size,), 99, device=x.device).long()
            #             noise1 = apply_conditioning(noise1, cond, 0)
            #             pred = self.student.model(noise1, cond, s, command)
            #             pred = apply_conditioning(pred, cond, 0)
            #             print("1 step from pure noise: " , nn.MSELoss()(pred, x).item())
            #
            #     print("full step from pure noise: ", nn.MSELoss()(x_psi, x).item())


            # if step == 0 or step == 5000:
            #     samples = to_np(x_theta)
            #     normed_observations = samples[:, :, :]
            #     observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')
            #     scaled_xy = normed_observations[:, :, 0:2]
            #     observations = np.concatenate([scaled_xy, observations[:, :, 2:]], axis=-1)
            #     savepath = 'train'
            #     name = 'x_theta_' + str(step)
            #     self.renderer.composite3(savepath, observations, name)

            L_distill = nn.MSELoss()(x_theta, x_psi)
            loss_S = L_distill # + L_adv
            self.optimizer.zero_grad()
            loss_S.backward()
            self.optimizer.step()

            # if self.step == 1:
            #     for name, param in self.student.named_parameters():
            #         if param.grad is not None:
            #             print(f"Gradient for {name}: {param.grad.abs().mean().item()}")
            #         else:
            #             print(f"No gradient for {name}")

            # loss_D = self.adversarial_loss_discriminator(x, x_theta.detach(), command)
            # self.d_optimizer.zero_grad()
            # loss_D.backward()
            # self.d_optimizer.step()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                self.save()

            if step % self.log_freq == 0:
                print(f"Step {step}: Student Loss = {loss_S.item():.4f}")
                # print(f"Step {step}: Student Loss = {loss_S.item():.4f}, Discriminator Loss = {loss_D.item():.4f}")

            if step % 500 == 0:
                # writer.add_scalar("Detailed/Adversarial_Loss_Student", L_adv.item(), step)
                # writer.add_scalar("Detailed/Adversarial_Loss_Discriminator", loss_D.item(), step)
                writer.add_scalar("Detailed/Distillation_Loss", L_distill.item(), step)

            # if self.step == 0 and self.sample_freq:
            #     self.render_reference(self.n_reference)

            if self.step and self.step % self.record_freq == 0:
                self.record_samples()

            self.step += 1
        writer.close()

    def save(self):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.student.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.bucket, logger.prefix, 'checkpoint')
        os.makedirs(savepath, exist_ok=True)
        if self.save_checkpoints:
            savepath = os.path.join(savepath, f'state_{self.step}.pt')
        else:
            savepath = os.path.join(savepath, 'state.pt')
        torch.save(data, savepath)
        logger.print(f'[ utils/training ] Saved model to {savepath}')

    def load(self):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.bucket, logger.prefix, f'checkpoint/state_5000.pt')
        # data = logger.load_torch(loadpath)
        data = torch.load(loadpath)

        # self.step = data['step']
        self.teacher.load_state_dict(data['model'])
        self.student.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:,None]

        ## [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.dataset.action_dim:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')
        scaled_xy = normed_observations[:, :, 0:2]
        observations = np.concatenate([scaled_xy, observations[:, :, 2:]], axis=-1)

        # from diffusion.datasets.preprocessing import blocks_cumsum_quat
        # # observations = conditions + blocks_cumsum_quat(deltas)
        # observations = conditions + deltas.cumsum(axis=1)

        #### @TODO: remove block-stacking specific stuff
        # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
        # observations = blocks_add_kuka(observations)
        ####

        savepath = os.path.join('images', f'sample-reference.png')
        self.renderer.composite(savepath, observations)

    def record_samples(self, batch_size=1, n_samples=4):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, self.device)
            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            commands = [[1, 1.5, 0, 0], [2, 1.5, 0, 0], [3, 1.5, 0, 0], [0, 1.5, 0, 0]]
            # commands = [[1, 0.8, 0, 0], [1, -0.8, 0, 0], [1, 0, 0.4, 0], [1, 0, 0, 0.8]]
            # commands = [[2, 0.8, 0, 0], [2, -0.8, 0, 0], [2, 0, 0.4, 0], [2, 0, 0, 0.8]]
            # commands = [[3, 0.8, 0, 0], [3, -0.8, 0, 0], [3, 0, 0.4, 0], [3, 0, 0, 0.8]]

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            if self.ema_model.returns_condition:
                # returns = to_device( 0.9 * torch.ones(n_samples, 1), self.device)
                ############################ change the gait here ######################################
                returns = to_device(torch.Tensor([commands[i]
                                                  for i in range(n_samples)]), self.device)
                #########################################################################################
            else:
                returns = None

            samples = self.ema_model.conditional_sample_acc(conditions, returns=returns)
            samples = to_np(samples)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = samples[:, :, :]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]

            # from diffusion.datasets.preprocessing import blocks_cumsum_quat
            # observations = conditions + blocks_cumsum_quat(deltas)
            # observations = conditions + deltas.cumsum(axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')
            scaled_xy = normed_observations[:, :, 0:2]
            observations = np.concatenate([scaled_xy, observations[:, :, 2:]], axis=-1)

            savepath = 'train'
            name = str(self.step)
            self.renderer.composite3(savepath, observations, name)