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

import torch
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
        n_reference=8,
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
        self.optimizer = torch.optim.AdamW(diffusion_model.parameters(), lr=train_lr)
        # self.optimizer = AdamP(diffusion_model.parameters(), lr=train_lr, betas=(0.9, 0.999), weight_decay=1e-2)

        self.bucket = bucket
        self.n_reference = n_reference

        self.reset_parameters()
        self.step = 0

        self.device = train_device
        # self.env = load_play_env(headless=False)
        self.env = None

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
                # if step % 20 == 0:
                #     diff_losses, inv_loss, _ = self.model.loss1(*batch)
                #     diff_loss = diff_losses.mean()
                #
                #     scaled_loss = np.sqrt(to_np(diff_losses)) / 2. * state_scale
                #
                #     writer.add_scalar("loss/diff loss", diff_loss, step)
                #     writer.add_scalar("loss/inv loss", inv_loss, step)

                if step %  2000 == 0:
                    diff_losses, inv_loss, x_targ, x_pred, a_targ, a_pred = self.model.loss2(*batch)
                    diff_loss = diff_losses.mean()

                    normed_loss = (to_np(x_targ) - to_np(x_pred)) ** 2
                    normed_loss = np.mean(normed_loss, axis=(0,1))
                    targ_unnormed = self.dataset.normalizer.unnormalize(to_np(x_targ), 'observations')
                    pred_unnormed = self.dataset.normalizer.unnormalize(to_np(x_pred), 'observations')
                    org_loss = (targ_unnormed - pred_unnormed) ** 2
                    org_loss = np.mean(org_loss, axis=(0,1))

                    a_unnormed = self.dataset.normalizer.unnormalize(to_np(a_targ), 'actions')
                    a_pred_unnormed = self.dataset.normalizer.unnormalize(to_np(a_pred), 'actions')
                    unnormed_inv_loss = (a_pred_unnormed - a_unnormed) ** 2
                    unnormed_inv_loss = np.mean(unnormed_inv_loss, axis=(0,1))

                    writer.add_scalar("loss/diff loss", diff_loss, step)
                    writer.add_scalar("loss/inv loss", inv_loss, step)
                    writer.add_scalar("loss/unnormed inv loss", unnormed_inv_loss, step)

                    writer.add_scalar("error/base_pos[m]", np.sqrt(np.sum(normed_loss[0:2])), step)
                    # writer.add_scalar("error/base_pos[m]", np.sqrt(np.sum(org_loss[0:3])), step)
                    writer.add_scalar("error/base_ori[quat]", np.sqrt(np.sum(org_loss[3:7])), step)
                    writer.add_scalar("error/base_lin_vel[m/s]", np.sqrt(np.sum(org_loss[7:10])), step)
                    writer.add_scalar("error/base_ang_vel[rad/s]", np.sqrt(np.sum(org_loss[10:13])), step)
                    writer.add_scalar("error/joint_pos[rad]", np.sqrt(np.sum(org_loss[18:30])), step)
                    writer.add_scalar("error/joint_vel[rad/s]", np.sqrt(np.sum(org_loss[30:42])), step)


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

            # if self.sample_freq and self.step % self.sample_freq == 0:
            #     if self.model.__class__ == diffuser.models.diffusion.GaussianInvDynDiffusion:
            #         self.inv_render_samples()
            #     elif self.model.__class__ == diffuser.models.diffusion.ActionGaussianDiffusion:
            #         pass
            #     else:
            #         self.render_samples()

            # if self.step == 0 or self.step % self.eval_freq == 0:
            #     warm = True if self.step % 2000 == 0 else False
            #     self.evaluate(warm)

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

    def record_samples(self, batch_size=2, n_samples=2):
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

            commands = [[1,1.0,0,0], [1,-1.0,0,0], [1,0,0.5,0], [1,0,0,1]]

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            if self.ema_model.returns_condition:
                # returns = to_device( 0.9 * torch.ones(n_samples, 1), self.device)
                ############################ change the gait here ######################################
                returns = to_device(torch.Tensor([random.choice(commands)
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
            scaled_xy = normed_observations[:, :, 0:2]
            observations = np.concatenate([scaled_xy, observations[:, :, 2:]], axis=-1)

            savepath = os.path.join('images', f'sample-{i}.png')
            name = str(self.step)
            self.renderer.composite2(savepath, observations, name)