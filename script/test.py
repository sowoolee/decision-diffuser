import isaacgym

assert isaacgym
import torch
import numpy as np

import glob
import pickle as pkl

from go1_gym.envs import *
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv

from config.locomotion_config import Config

from tqdm import tqdm

import random
import time

import os
from copy import deepcopy
from diffuser.utils.arrays import to_torch, to_np, to_device

def load_env(label, headless=False):
    dirs = glob.glob(f"../runs/{label}/*")
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
    Cfg.terrain.num_rows = 20
    Cfg.terrain.num_cols = 20
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True

    Cfg.domain_rand.lag_timesteps = 1
    Cfg.domain_rand.randomize_lag_timesteps = False
    Cfg.control.control_type = "P"

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=Cfg)

    return env


def batch_to_device(batch, device):
    vals = [
        to_device(getattr(batch, field), device)
        for field in batch._fields
    ]
    return type(batch)(*vals)

def cycle(dl):
    while True:
        for data in dl:
            yield data


def import_diffuser():
    import diffuser.utils as utils
    from ml_logger import logger, RUN
    from config.locomotion_config import Config

    # logger.remove('*.pkl')
    # logger.remove("traceback.err")
    # logger.log_params(Config=vars(Config), RUN=vars(RUN))

    Config.device = 'cuda:0'

    loadpath = '/home/hubolab/workspace/DD/weights/diffuser/go1_locomotion/mamba/checkpoint'
    loadpath = os.path.join(loadpath, 'state_4000.pt')
    state_dict = torch.load(loadpath, map_location=Config.device)

    # Load configs
    torch.backends.cudnn.benchmark = True
    utils.set_seed(Config.seed)

    dataset_config = utils.Config(
        Config.loader,
        savepath='dataset_config.pkl',
        env=Config.dataset,
        horizon=Config.horizon,
        normalizer=Config.normalizer,
        preprocess_fns=Config.preprocess_fns,
        use_padding=Config.use_padding,
        max_path_length=Config.max_path_length,
        include_returns=Config.include_returns,
        returns_scale=Config.returns_scale,
    )

    render_config = utils.Config(
        Config.renderer,
        savepath='render_config.pkl',
        env=Config.dataset,
    )

    dataset = dataset_config()
    renderer = render_config()

    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim

    if Config.diffusion == 'models.GaussianInvDynDiffusion':
        transition_dim = observation_dim
    else:
        transition_dim = observation_dim + action_dim

    model_config = utils.Config(
        Config.model,
        savepath='model_config.pkl',
        horizon=Config.horizon,
        transition_dim=transition_dim,
        cond_dim=observation_dim,
        dim_mults=Config.dim_mults,
        dim=Config.dim,
        returns_condition=Config.returns_condition,
        device=Config.device,
    )

    diffusion_config = utils.Config(
        Config.diffusion,
        savepath='diffusion_config.pkl',
        horizon=Config.horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        n_timesteps=Config.n_diffusion_steps,
        loss_type=Config.loss_type,
        clip_denoised=Config.clip_denoised,
        predict_epsilon=Config.predict_epsilon,
        hidden_dim=Config.hidden_dim,
        ## loss weighting
        action_weight=Config.action_weight,
        loss_weights=Config.loss_weights,
        loss_discount=Config.loss_discount,
        returns_condition=Config.returns_condition,
        device=Config.device,
        condition_guidance_w=Config.condition_guidance_w,
    )

    trainer_config = utils.Config(
        utils.Trainer,
        savepath='trainer_config.pkl',
        train_batch_size=Config.batch_size,
        train_lr=Config.learning_rate,
        gradient_accumulate_every=Config.gradient_accumulate_every,
        ema_decay=Config.ema_decay,
        sample_freq=Config.sample_freq,
        save_freq=Config.save_freq,
        log_freq=Config.log_freq,
        label_freq=int(Config.n_train_steps // Config.n_saves),
        save_parallel=Config.save_parallel,
        bucket=Config.bucket,
        n_reference=Config.n_reference,
        train_device=Config.device,
    )

    model = model_config()
    diffusion = diffusion_config(model)
    trainer = trainer_config(diffusion, dataset, renderer)
    logger.print(utils.report_parameters(model), color='green')
    trainer.step = state_dict['step']
    trainer.model.load_state_dict(state_dict['model'])
    trainer.ema_model.load_state_dict(state_dict['ema'])

    assert trainer.ema_model.condition_guidance_w == Config.condition_guidance_w

    return trainer


def test():
    label = "gait-conditioned-agility/pretrain-v0/train"
    env = load_env(label, headless=False)

    # import diffusion model
    trainer = import_diffuser()
    dataset = trainer.dataset
    device = trainer.device

    # load environment
    num_envs = env.num_envs

    # y conditioning
    gait_num = 1
    v_x = 1.5
    returns = to_device(torch.Tensor([[gait_num, v_x, 0,0] for i in range(num_envs)]), device)

    # start testing
    t = 0
    env.reset()
    total_steps = 200

    while t < total_steps:
        obs = np.concatenate([
            to_np([[0.,0.]]),
            to_np(env.root_states[:,2:3]), to_np(env.root_states[:,3:7]),
            to_np(env.root_states[:,7:10]), to_np(env.root_states[:,10:13]),
            to_np(env.dof_pos[:,:12]), to_np(env.dof_vel[:, :12])], axis=-1)

        # action sampling
        obs = dataset.normalizer.normalize(obs, 'observations')
        obs = np.concatenate([to_np([[0.,0.]]), obs[:,2:]], axis=-1)

        conditions = {0: to_torch(obs, device=device)}

        # state trajectory sampling
        samples = trainer.ema_model.conditional_sample(conditions, returns)
        obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)

        with torch.no_grad():
            action = trainer.ema_model.inv_model(obs_comb)
            env.step(action)
            env.set_camera(env.root_states[0, 0:3] + to_torch([2.5, 2.5, 2.5]), env.root_states[0, 0:3])

        print("Environment timestep: {}".format(t))

        t += 1

    print('evaluation ended')

if __name__ == '__main__':
    test()