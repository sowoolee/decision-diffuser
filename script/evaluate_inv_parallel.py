import isaacgym
assert isaacgym

import diffuser.utils as utils
from ml_logger import logger
import torch
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
import os
import gym
from config.locomotion_config import Config
from diffuser.utils.arrays import to_torch, to_np, to_device

import glob
import pickle as pkl
from go1_gym.envs import *
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv
import time

torch.cuda.set_per_process_memory_fraction(fraction=0.4, device=0)
from tqdm import tqdm

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

def load_env(render=False, headless=False):
    # prepare environment
    config_go1(Cfg)

    Cfg.commands.num_lin_vel_bins = 30
    Cfg.commands.num_ang_vel_bins = 30
    Cfg.curriculum_thresholds.tracking_ang_vel = 0.7
    Cfg.curriculum_thresholds.tracking_lin_vel = 0.8
    Cfg.curriculum_thresholds.tracking_contacts_shaped_vel = 0.9
    Cfg.curriculum_thresholds.tracking_contacts_shaped_force = 0.9

    Cfg.commands.distributional_commands = True

    Cfg.env.priv_observe_motion = False
    Cfg.env.priv_observe_gravity_transformed_motion = False #
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.env.priv_observe_friction_indep = False
    Cfg.domain_rand.randomize_friction = True
    Cfg.env.priv_observe_friction = False
    Cfg.domain_rand.friction_range = [0.0, 0.0]
    Cfg.domain_rand.randomize_restitution = True
    Cfg.env.priv_observe_restitution = False
    Cfg.domain_rand.restitution_range = [0.0, 1.0]
    Cfg.domain_rand.randomize_base_mass = True
    Cfg.env.priv_observe_base_mass = False
    Cfg.domain_rand.added_mass_range = [-1.0, 3.0]
    Cfg.domain_rand.randomize_gravity = True
    Cfg.domain_rand.gravity_range = [-1.0, 1.0]
    Cfg.domain_rand.gravity_rand_interval_s = 2.0
    Cfg.domain_rand.gravity_impulse_duration = 0.5
    Cfg.env.priv_observe_gravity = False #
    Cfg.domain_rand.randomize_com_displacement = False
    Cfg.domain_rand.com_displacement_range = [-0.15, 0.15]
    Cfg.env.priv_observe_com_displacement = False
    Cfg.domain_rand.randomize_ground_friction = True
    Cfg.env.priv_observe_ground_friction = False
    Cfg.env.priv_observe_ground_friction_per_foot = False
    Cfg.domain_rand.ground_friction_range = [0.3, 2.0]
    Cfg.domain_rand.randomize_motor_strength = True
    Cfg.domain_rand.motor_strength_range = [0.9, 1.1]
    Cfg.env.priv_observe_motor_strength = False
    Cfg.domain_rand.randomize_motor_offset = True
    Cfg.domain_rand.motor_offset_range = [-0.02, 0.02]
    Cfg.env.priv_observe_motor_offset = False
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.env.priv_observe_Kp_factor = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.env.priv_observe_Kd_factor = False
    Cfg.env.priv_observe_body_velocity = False
    Cfg.env.priv_observe_body_height = False
    Cfg.env.priv_observe_desired_contact_states = False
    Cfg.env.priv_observe_contact_forces = False
    Cfg.env.priv_observe_foot_displacement = False

    Cfg.env.num_privileged_obs = 0 #3
    Cfg.env.num_observation_history = 30
    Cfg.reward_scales.feet_contact_forces = 0.0

    Cfg.domain_rand.rand_interval_s = 4
    Cfg.commands.num_commands = 15
    Cfg.env.observe_two_prev_actions = False #
    Cfg.env.observe_yaw = False #
    Cfg.env.num_observations = 71
    Cfg.env.num_scalar_observations = 71
    Cfg.env.observe_gait_commands = False #
    Cfg.env.observe_timing_parameter = False
    Cfg.env.observe_clock_inputs = True #

    Cfg.domain_rand.tile_height_range = [-0.0, 0.0]
    Cfg.domain_rand.tile_height_curriculum = False
    Cfg.domain_rand.tile_height_update_interval = 1000, 3000
    Cfg.domain_rand.tile_height_curriculum_step = 0.01
    Cfg.terrain.border_size = 0.0

    Cfg.commands.resampling_time = 10

    Cfg.reward_scales.feet_slip = -0.04
    Cfg.reward_scales.action_smoothness_1 = -0.1
    Cfg.reward_scales.action_smoothness_2 = -0.1
    Cfg.reward_scales.dof_vel = -1e-4
    Cfg.reward_scales.dof_pos = -0.05
    Cfg.reward_scales.jump = 10.0
    Cfg.reward_scales.base_height = 0.0
    Cfg.rewards.base_height_target = 0.30
    Cfg.reward_scales.estimation_bonus = 0.0

    Cfg.reward_scales.feet_impact_vel = -0.0

    # rewards.footswing_height = 0.09
    Cfg.reward_scales.feet_clearance = -0.0
    Cfg.reward_scales.feet_clearance_cmd = -15.

    # reward_scales.feet_contact_forces = -0.01

    Cfg.rewards.reward_container_name = "CoRLRewards"
    Cfg.rewards.only_positive_rewards = False
    Cfg.rewards.only_positive_rewards_ji22_style = True
    Cfg.rewards.sigma_rew_neg = 0.02

    Cfg.reward_scales.hop_symmetry = 0.0
    Cfg.rewards.kappa_gait_probs = 0.07
    Cfg.rewards.gait_force_sigma = 100.
    Cfg.rewards.gait_vel_sigma = 10.

    Cfg.reward_scales.tracking_contacts_shaped_force = 4.0
    Cfg.reward_scales.tracking_contacts_shaped_vel = 4.0

    Cfg.reward_scales.collision = -5.0

    Cfg.commands.lin_vel_x = [-1.0, 1.0]
    Cfg.commands.lin_vel_y = [-0.6, 0.6]
    Cfg.commands.ang_vel_yaw = [-1.0, 1.0]
    Cfg.commands.body_height_cmd = [-0.25, 0.15]
    Cfg.commands.gait_frequency_cmd_range = [1.5, 4.0]
    Cfg.commands.gait_phase_cmd_range = [0.0, 1.0]
    Cfg.commands.gait_offset_cmd_range = [0.0, 1.0]
    Cfg.commands.gait_bound_cmd_range = [0.0, 1.0]
    Cfg.commands.gait_duration_cmd_range = [0.5, 0.5]
    Cfg.commands.footswing_height_range = [0.03, 0.25]

    Cfg.reward_scales.lin_vel_z = -0.02
    Cfg.reward_scales.ang_vel_xy = -0.001
    Cfg.reward_scales.base_height = 0.0
    Cfg.reward_scales.feet_air_time = 0.0

    Cfg.commands.limit_vel_x = [-5.0, 5.0]
    Cfg.commands.limit_vel_y = [-0.6, 0.6]
    Cfg.commands.limit_vel_yaw = [-5.0, 5.0]
    Cfg.commands.limit_body_height = [-0.25, 0.15]
    Cfg.commands.limit_gait_frequency = [1.5, 4.0]
    Cfg.commands.limit_gait_phase = [0.0, 1.0]
    Cfg.commands.limit_gait_offset = [0.0, 1.0]
    Cfg.commands.limit_gait_bound = [0.0, 1.0]
    Cfg.commands.limit_gait_duration = [0.5, 0.5]
    Cfg.commands.limit_footswing_height = [0.03, 0.25]

    Cfg.commands.num_bins_vel_x = 21
    Cfg.commands.num_bins_vel_y = 1
    Cfg.commands.num_bins_vel_yaw = 21
    Cfg.commands.num_bins_body_height = 1
    Cfg.commands.num_bins_gait_frequency = 1
    Cfg.commands.num_bins_gait_phase = 1
    Cfg.commands.num_bins_gait_offset = 1
    Cfg.commands.num_bins_gait_bound = 1
    Cfg.commands.num_bins_gait_duration = 1
    Cfg.commands.num_bins_footswing_height = 1

    Cfg.normalization.friction_range = [0, 1]
    Cfg.normalization.ground_friction_range = [0, 1]

    Cfg.commands.exclusive_phase_offset = False
    Cfg.commands.pacing_offset = False
    Cfg.commands.binary_phases = True
    Cfg.commands.gaitwise_curricula = True

    # 5 times per second

    Cfg.env.num_envs = 1
    Cfg.domain_rand.push_interval_s = 1
    Cfg.terrain.num_rows = 3
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    Cfg.domain_rand.randomize_friction = True
    Cfg.domain_rand.friction_range = [1.0, 1.01]
    Cfg.domain_rand.randomize_base_mass = True
    Cfg.domain_rand.added_mass_range = [0., 6.]
    Cfg.terrain.terrain_noise_magnitude = 0.0
    # Cfg.asset.fix_base_link = True

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "actuator_net"

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=False, cfg=Cfg)
    return env


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
    Cfg.control.control_type = "P"

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=Cfg)

    return env


def evaluate(**deps):
    from ml_logger import logger, RUN
    from config.locomotion_config import Config

    RUN._update(deps)
    Config._update(deps)

    logger.remove('*.pkl')
    logger.remove("traceback.err")
    logger.log_params(Config=vars(Config), RUN=vars(RUN))

    Config.device = 'cuda'

    if Config.predict_epsilon:
        prefix = f'predict_epsilon_{Config.n_diffusion_steps}_1000000.0'
    else:
        prefix = f'predict_x0_{Config.n_diffusion_steps}_1000000.0'

    loadpath = os.path.join(Config.bucket, logger.prefix, 'checkpoint')
    
    if Config.save_checkpoints:
        loadpath = os.path.join(loadpath, f'state_{self.step}.pt')
    else:
        loadpath = os.path.join(loadpath, 'state.pt')
    
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

    num_eval = 1
    device = Config.device

    env_list = []

    for evaluation in range(num_eval):
        env = load_play_env(headless=False) # play env version
        # env = load_env(render=True, headless=False) # test env version
        env_list.append(env)

    dones = [0 for _ in range(num_eval)]
    episode_rewards = [0 for _ in range(num_eval)]

    assert trainer.ema_model.condition_guidance_w == Config.condition_guidance_w
    returns = to_device(Config.test_ret * torch.ones(num_eval, 1), device)

    default_pos = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5])

    t = 0
    # ############### test env ver ######################
    # # obs_list = [env.reset()[None] for env in env_list]
    # obs_list = [env.reset().detach().cpu() for env in env_list]
    # obs = np.concatenate(obs_list, axis=0)
    # q = env.root_states[0, 3:7].detach().cpu()
    # q = np.array(q[[3, 0, 1, 2]][None])
    # clock = env.clock_inputs[0,:].detach().cpu()
    # clock = np.array(clock)
    # obs = np.concatenate([obs[:,:3], obs[:,42:46], obs[:,18:30]+default_pos, obs[:,30:42]], axis=-1)
    ####################################################3

    ################# play env ver #######################
    obs_list = [env.reset().detach().cpu() for env in env_list]
    obs = np.concatenate(obs_list, axis=0)
    obs = np.concatenate([obs[:, :3], obs[:, 66:70], obs[:, 18:30] + default_pos, obs[:, 30:42]], axis=-1)
    ##########################################################

    recorded_obs = [deepcopy(obs[:, None])]

    # while sum(dones) <  num_eval:
    while t < 100:
        obs = dataset.normalizer.normalize(obs, 'observations')
        conditions = {0: to_torch(obs, device=device)}
        samples = trainer.ema_model.conditional_sample(conditions, returns=returns)
        obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)
        obs_comb = obs_comb.reshape(-1, 2*observation_dim)
        action = trainer.ema_model.inv_model(obs_comb)

        samples = to_np(samples)
        action = to_np(action)

        action = dataset.normalizer.unnormalize(action, 'actions')
        action = to_torch(action[None])

        if t == 0:
            normed_observations = samples[:, :, :]
            observations = dataset.normalizer.unnormalize(normed_observations, 'observations')
            savepath = os.path.join('images', 'sample-planned.png')
            renderer.composite(savepath, observations)

        obs_list = []
        for i in range(num_eval):
            this_obs, this_reward, this_done, _ = env_list[i].step(action[i])
            this_obs = this_obs.detach().cpu()

            ######################### test env ver ##########################################
            # this_obs = np.concatenate([this_obs[:, :3], this_obs[:, 42:46],
            #                            this_obs[:, 18:30] + default_pos, this_obs[:, 30:42]], axis=-1) # gravity clock jointpos jointvel
            ##################################################################################
            this_obs = np.concatenate([this_obs[:, :3], this_obs[:, 66:70],
                                        this_obs[:, 18:30] + default_pos, this_obs[:, 30:42]], axis=-1) #
            ######################### play env ver ###########################################

            ##################################################################################
            obs_list.append(this_obs)
            if this_done:
                if dones[i] == 1:
                    pass
                else:
                    dones[i] = 1
                    # episode_rewards[i] += this_reward
                    episode_rewards[i] += 0.1
                    logger.print(f"Episode ({i}): {episode_rewards[i]}", color='green')
            else:
                if dones[i] == 1:
                    pass
                else:
                    # episode_rewards[i] += this_reward
                    episode_rewards[i] += 0.1

        obs = np.concatenate(obs_list, axis=0)
        recorded_obs.append(deepcopy(obs[:, None]))
        t += 1

    recorded_obs = np.concatenate(recorded_obs, axis=1)
    savepath = os.path.join('images', f'sample-executed.png')
    renderer.composite1(savepath, recorded_obs)
    episode_rewards = np.array(episode_rewards)

    logger.print(f"average_ep_reward: {np.mean(episode_rewards)}, std_ep_reward: {np.std(episode_rewards)}", color='green')
    logger.log_metrics_summary({'average_ep_reward':np.mean(episode_rewards), 'std_ep_reward':np.std(episode_rewards)})


def evaluate_fast(**deps):
    from ml_logger import logger, RUN
    from config.locomotion_config import Config

    RUN._update(deps)
    Config._update(deps)

    logger.remove('*.pkl')
    logger.remove("traceback.err")
    logger.log_params(Config=vars(Config), RUN=vars(RUN))

    Config.device = 'cuda:1'

    if Config.predict_epsilon:
        prefix = f'predict_epsilon_{Config.n_diffusion_steps}_1000000.0'
    else:
        prefix = f'predict_x0_{Config.n_diffusion_steps}_1000000.0'

    loadpath = os.path.join(Config.bucket, logger.prefix, 'checkpoint')

    if Config.save_checkpoints:
        loadpath = os.path.join(loadpath, f'state_{self.step}.pt')
    else:
        loadpath = os.path.join(loadpath, 'state.pt')

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

    num_eval = 1
    device = Config.device

    gaits = {"pronking": [0, 0, 0],
             "trotting": [0.5, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, 0.5]}

    env_list = []

    for evaluation in range(num_eval):
        env = load_play_env(headless=False)  # play env version
        # env = load_env(render=True, headless=False) # test env version
        env_list.append(env)

    dones = [0 for _ in range(num_eval)]
    episode_rewards = [0 for _ in range(num_eval)]
    sampling_time = 0

    assert trainer.ema_model.condition_guidance_w == Config.condition_guidance_w
    ####################### return version #############################
    # returns = to_device(Config.test_ret * torch.ones(num_eval, 1), device)
    #######################  gait version  #############################
    gait_idx = 3
    gait_indices = [gait_idx for _ in range(num_eval)]  # just for one gait
    random_gaits = [list(gaits.values())[idx] for idx in gait_indices]
    gait = torch.tensor(random_gaits)
    step_frequency = 3.0

    returns = to_device(torch.Tensor([[gait_idx,1.5,0,0] for i in range(num_eval)]), device)

    # env_list[0].commands[:, 0] = 0.
    # env_list[0].commands[:, 1] = 0.
    # env_list[0].commands[:, 2] = 0.
    # env_list[0].commands[:, 3] = 0.
    # env_list[0].commands[:, 4] = step_frequency
    # env_list[0].commands[:, 5:8] = gait
    # env_list[0].commands[:, 8] = 0.5
    # env_list[0].commands[:, 9] = 0.08
    # env_list[0].commands[:, 10] = 0.
    # env_list[0].commands[:, 11] = 0.
    # env_list[0].commands[:, 12] = 0.25
    #####################################################################

    default_pos = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5])

    ######################## for verifying torque ############################
    state_scale = dataset.normalizer.unnormalize(5*np.ones(41),'observations') - dataset.normalizer.unnormalize(-5*np.ones(41), 'observations')

    dataloader = cycle(torch.utils.data.DataLoader(
            dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
    batch = next(dataloader)
    batch = batch_to_device(batch, device=device)

    x = batch[0]
    cond = batch[1]
    returns = batch[2]

    batch_size = len(x)
    t = torch.randint(99, 100, (batch_size,), device=x.device).long()
    _, _, targ, pred = trainer.ema_model.p_losses1(x[:, :, action_dim:], cond, t, returns)

    targ_unnormed = trainer.dataset.normalizer.unnormalize(to_np(targ), 'observations')
    pred_unnormed = trainer.dataset.normalizer.unnormalize(to_np(pred), 'observations')
    org_loss = (targ_unnormed - pred_unnormed) ** 2
    org_loss = np.mean(org_loss, axis=(0, 1))

    # diffuse_loss = diffuse_loss.mean()
    # # Calculating inv loss
    # x_t = x[:, :-1, action_dim:]
    # a_t = x[:, :-1, :action_dim]
    # x_t_1 = x[:, 1:, action_dim:]
    # x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
    # x_comb_t = x_comb_t.reshape(-1, 2 * observation_dim)
    # a_t = a_t.reshape(-1, action_dim)
    #
    # pred_a_t = trainer.ema_model.inv_model(x_comb_t)
    # inv_loss = F.mse_loss(pred_a_t, a_t)


    ############  inv loss sanity check  ##############################
    # x = batch[0]
    # states = x[:, :, action_dim:]
    # actions = x[:, :, :action_dim]
    #
    # obs_comb = torch.cat([states[:, 4, :], states[:, 5, :]], dim=-1)
    # obs_comb = obs_comb.reshape(-1, 2 * observation_dim)
    # action_pred = trainer.ema_model.inv_model(obs_comb)
    #
    # action_pred = to_np(action_pred)
    # action = to_np(actions[:,4,:])
    #
    # action_pred = dataset.normalizer.unnormalize(action_pred, 'actions')
    # action_true = dataset.normalizer.unnormalize(action, 'actions')
    #
    # torques = 10.0 * 0.25 * (action_true[:, :12] - action_pred[:, :12])
    ###################################################################

    t = 0
    replay = False

    # env.reset()
    ################ can evalutate just one env ##################
    ################ play env obdim 42 ver #######################
    for env in env_list:
        obs_list = []

        file_path = os.path.expanduser('~/Desktop/data.pkl')
        with open(file_path, 'rb') as f:
            loaded_data = pkl.load(f)
            data = loaded_data

        env.reset()
        obs = dataset.normalizer.unnormalize(to_np(cond[0]), 'observations')


        env_ids = torch.tensor([0], dtype=torch.int32, device=device)
        # dof_pos = to_torch(default_pos)
        # base_state = to_torch([[env.root_states[0,0], env.root_states[0,1], 0.3, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        dof_pos = to_torch(obs[:,17:29])
        dof_vel = to_torch(obs[:,29:41])
        base_state = to_torch([[env.root_states[0,0], env.root_states[0,1], obs[0,2], *obs[0,3:7], *obs[0,7:13]]])
        base_state = to_torch([obs[0, 0:13]])

        env.set_idx_state(env_ids, dof_pos, dof_vel, base_state)

        if replay:
            env_ids = torch.tensor([0]).to('cuda:0')
            obs = torch.Tensor([data['observations'][0]]).to('cuda:0')
            dof_pos = torch.Tensor(obs[:,17:29]).to('cuda:0')
            dof_vel = torch.Tensor(obs[:,29:41]).to('cuda:0')
            base_state = torch.Tensor([[env.root_states[0,0], env.root_states[0,1], obs[0,2], *obs[0,3:7], *obs[0,7:13]]]).to('cuda:0')

            env.set_idx_state(env_ids, dof_pos, dof_vel, base_state)

        # env.set_camera(env.root_states[0, 0:3] + to_torch([2.5, 2.5, 2.5]), env.root_states[0, 0:3])

        ############## true action check #################################
        # for T in range(55):
        #     env_ids = torch.tensor([0], dtype=torch.int32, device='cuda:0')
        #     x_t = to_np(x[:, T, action_dim:])
        #     x_t = dataset.normalizer.unnormalize(x_t, 'observations')
        #
        #     # dof_pos = to_torch(x_t[:,17:29], device='cuda:0')
        #     # dof_vel = to_torch(x_t[:,29:41], device='cuda:0')
        #     # base_state = to_torch([[env.root_states[0,0], env.root_states[0,1], x_t[0,2], *x_t[0,3:7], *x_t[0,7:13]]], device='cuda:0')
        #     # env.set_idx_state(env_ids, dof_pos, dof_vel, base_state)
        #     # env.set_camera(env.root_states[0, 0:3] + to_torch([2.5, 2.5, 2.5]), env.root_states[0, 0:3])
        #
        #     a_t = x[:, T, :action_dim]
        #     a_t = to_np(a_t)
        #     a_t = dataset.normalizer.unnormalize(a_t, 'actions')
        #
        #     env.step(to_torch(a_t))
        #     # acts = torch.Tensor([data['actions'][T]])
        #     # env.step(acts)
        #     # time.sleep(0.1)
        ##################################################

        # for i in range(1):
        #     env.commands[:, 4] = step_frequency
        #     env.commands[:, 5:8] = gait
        #     obs, _, _, _ = env.step(torch.zeros(size=(1,12), device=device))

        env.commands[:, 4] = step_frequency
        env.commands[:, 5:8] = gait
        # obs_list.append(obs.detach().cpu())
        # obs_list.append(env.reset().detach().cpu())
        init_xy = env.root_states[:, 0:2]
        # obs = np.concatenate(obs_list, axis=0)
        obs = np.concatenate([
                                    to_np([[0.,0.]]),
                                    to_np(env.root_states[:,2:3]), to_np(env.root_states[:,3:7]),
                                    to_np(env.root_states[:,7:10]), to_np(env.base_ang_vel[:,:]),
                                    to_np(env.clock_inputs),
                                    to_np(env.dof_pos[:,:12]), to_np(env.dof_vel[:, :12])], axis=-1)
                                    # obs[:, 66:70],  # clock inputs
                                    # obs[:, 18:30] + default_pos, obs[:, 30:42] * 20], axis=-1)
    ##############################################################

    recorded_obs = [deepcopy(obs[:, None])]
    warm_sample = None
    warm = False

    # while sum(dones) <  num_eval:
    while t < 150:
        obs = dataset.normalizer.normalize(obs, 'observations')
        conditions = {0: to_torch(obs, device=device)}

        if not warm:
            samples = trainer.ema_model.conditional_sample(conditions, returns)

        elif warm:
            if t == 0 :
                with torch.no_grad():
                    samples = trainer.ema_model.conditional_sample(conditions, returns)
                # torch.save(samples, 'initial_sample_pace.pt')
                # samples = torch.load('initial_sample.pt')
                    warm_sample = samples
            ############################# mixing gait ##############################################
            else :
                if t < 500:
                    start_time = time.time()
                    with torch.no_grad():
                        samples = trainer.ema_model.conditional_warm_sample(conditions, warm_sample, k=1, steps=10, returns=returns)
                    end_time = time.time()
                    execution_time = end_time - start_time
                    sampling_time += execution_time
                    warm_sample = samples
            ########################################################################################
        # if t==0:
        #     samples = trainer.ema_model.conditional_sample(cond, returns)


        # if t % 5 == 0:
        #     actions = []
        #     for i in range(5):
        #         obs_comb = obs_comb = torch.cat([samples[:, i, :], samples[:, i + 1, :]], dim=-1)
        #         obs_comb = obs_comb.reshape(-1, 2 * observation_dim)
        #         action = trainer.ema_model.inv_model(obs_comb)
        #         action = to_np(action)
        #         action = dataset.normalizer.unnormalize(action, 'actions')
        #         action = to_torch(action[None])
        #         actions.append(action)

        obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)
        obs_comb = obs_comb.reshape(-1, 2 * observation_dim)
        action = trainer.ema_model.inv_model(obs_comb)

        samples = to_np(samples)
        action = to_np(action)

        action = dataset.normalizer.unnormalize(action, 'actions')
        action = to_torch(action[None])

        if t == 0:
            normed_observations = samples[:, :, :]
            observations = dataset.normalizer.unnormalize(normed_observations, 'observations')
            savepath = os.path.join('images', 'sample-planned.png')
            renderer.composite2(savepath, observations, str(t))

        obs_list = []
        for i in range(num_eval):
            with torch.no_grad():
                env_list[i].commands[:, 4] = step_frequency
                env_list[i].commands[:, 5:8] = gait

                this_obs, this_reward, this_done, _ = env_list[i].step(action[i])
                # this_obs, this_reward, this_done, _ = env_list[i].step(actions[ t % 5 ][i])
                env_list[i].set_camera(env_list[i].root_states[0, 0:3] + to_torch([2.5, 2.5, 2.5]), env_list[i].root_states[0, 0:3])
            this_obs = this_obs.detach().cpu()

            env_list[i].commands[:, 4] = step_frequency
            env_list[i].commands[:, 5:8] = gait
            ######################### play env obdim 42 ver ##################################
            this_obs = np.concatenate([ to_np([[0.,0.]]), to_np(env_list[i].root_states[:,2:3]), to_np(env_list[i].root_states[:,3:7]),
                                            to_np(env_list[i].root_states[:,7:10]), to_np(env_list[i].base_ang_vel[:,:]),
                                            to_np(env_list[i].clock_inputs),
                                            to_np(env_list[i].dof_pos[:,:12]), to_np(env_list[i].dof_vel[:,:12])], axis=-1)
                                            # this_obs[:, 66:70],
                                            # this_obs[:, 18:30] + default_pos, this_obs[:, 30:42] * 20], axis=-1)
            ##################################################################################
            obs_list.append(this_obs)
            if this_done:
                if dones[i] == 1:
                    pass
                else:
                    dones[i] = 1
                    # episode_rewards[i] += this_reward.detach().cpu()
                    episode_rewards[i] += env_list[i].base_lin_vel[0,0].cpu() / 15.
                    logger.print(f"Episode ({i}): {episode_rewards[i]}", color='green')
            else:
                if dones[i] == 1:
                    pass
                else:
                    # episode_rewards[i] += this_reward.detach().cpu()
                    episode_rewards[i] += env_list[i].base_lin_vel[0,0].cpu() / 15.
                    # episode_rewards[i] += torch.norm(env_list[i].base_lin_vel[0, 0:2], 2).cpu() / 15.

        obs = np.concatenate(obs_list, axis=0)
        recorded_obs.append(deepcopy(obs[:, None]))
        t += 1

    print("###################### end of simulation ######################")
    print(len(recorded_obs))

    clock_true = np.zeros(151)
    clock_inputs = np.zeros(151)
    for i, obs in enumerate(recorded_obs):
        clock_inputs[i] = obs[0][0][13]
        # env_ids = torch.tensor([0], dtype=torch.int32, device=device)
        # dof_pos =  to_torch(obs[0,:,17:29])
        # base_state = to_torch(obs[0,:,0:13])
        # env_list[0].set_camera(base_state[0,0:3]+to_torch([0, 0, 2.5]),base_state[0,0:3])
        # env_list[0].set_idx_pose(env_ids, dof_pos, base_state)
        # time.sleep(0.05)
    for i in range(56):
        clock_true[i] = targ_unnormed[0][i][13]
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(12, 5))

    axs[0].plot(np.linspace(0, 151 * env_list[0].dt, 151), clock_true, linestyle="-", label="Measured")
    axs[0].set_title("True Clock inputs")

    axs[1].plot(np.linspace(0, 151 * env_list[0].dt, 151), clock_inputs, linestyle="-", label="Measured")
    axs[1].set_title("Clock inputs")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Phase")

    plt.tight_layout()
    plt.show()

    print("코드 실행 시간:", sampling_time, "초")

    recorded_obs = np.concatenate(recorded_obs, axis=1)
    savepath = os.path.join('images', f'sample-executed.png')
    renderer.composite2(savepath, recorded_obs, 'rollout')
    episode_rewards = np.array(episode_rewards)

    logger.print(f"average_ep_reward: {np.mean(episode_rewards)}, std_ep_reward: {np.std(episode_rewards)}",
                 color='green')
    logger.log_metrics_summary(
        {'average_ep_reward': np.mean(episode_rewards), 'std_ep_reward': np.std(episode_rewards)})


def evaluate_both(**deps):
    from ml_logger import logger, RUN
    from config.locomotion_config import Config

    RUN._update(deps)
    Config._update(deps)

    logger.remove('*.pkl')
    logger.remove("traceback.err")
    logger.log_params(Config=vars(Config), RUN=vars(RUN))

    Config.device = 'cuda:1'

    if Config.predict_epsilon:
        prefix = f'predict_epsilon_{Config.n_diffusion_steps}_1000000.0'
    else:
        prefix = f'predict_x0_{Config.n_diffusion_steps}_1000000.0'

    loadpath = os.path.join(Config.bucket, logger.prefix, 'checkpoint')

    if Config.save_checkpoints:
        loadpath = os.path.join(loadpath, f'state_{self.step}.pt')
    else:
        loadpath = os.path.join(loadpath, 'state.pt')

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

    num_eval = 1
    device = Config.device

    gaits = {"pronking": [0, 0, 0],
             "trotting": [0.5, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, 0.5]}

    env = load_play_env(headless=False)


    dones = [0 for _ in range(num_eval)]
    episode_rewards = [0 for _ in range(num_eval)]
    sampling_time = 0

    assert trainer.ema_model.condition_guidance_w == Config.condition_guidance_w

    #######################  gait version  #############################
    gait_idx = 1
    gait_indices = [gait_idx for _ in range(num_eval)]  # just for one gait
    random_gaits = [list(gaits.values())[idx] for idx in gait_indices]
    gait = torch.tensor(random_gaits)
    step_frequency = 3.0

    returns = to_device(torch.Tensor([[gait_idx,1.5,0,0] for i in range(num_eval)]), device)
    #####################################################################

    default_pos = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5])

    ######################## for verifying torque ############################
    state_scale = (dataset.normalizer.unnormalize(np.ones(dataset.observation_dim),'observations')
                   - dataset.normalizer.unnormalize(np.ones(dataset.observation_dim), 'observations'))

    dataloader = cycle(torch.utils.data.DataLoader(
            dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
    batch = next(dataloader)
    batch = batch_to_device(batch, device=device)

    # bring batch data
    x = batch[0]
    cond = batch[1]
    # returns = batch[2]

    batch_size = len(x)
    t = torch.randint(99, 100, (batch_size,), device=x.device).long()
    _, _, targ, pred = trainer.ema_model.p_losses1(x[:, :, action_dim:], cond, t, returns)

    targ_unnormed = trainer.dataset.normalizer.unnormalize(to_np(targ), 'observations')
    pred_unnormed = trainer.dataset.normalizer.unnormalize(to_np(pred), 'observations')


    t = 0
    replay = False

    # batch state setting
    env.reset()
    obs = dataset.normalizer.unnormalize(to_np(cond[0]), 'observations')

    # set batch state
    env_ids = torch.tensor([0], dtype=torch.int32, device=device)
    dof_pos = to_torch(obs[:,18:30])
    dof_vel = to_torch(obs[:,30:42])
    base_state = to_torch([obs[0, 0:13]])
    env.set_idx_state(env_ids, dof_pos, dof_vel, base_state)
    env.set_camera(env.root_states[0, 0:3] + to_torch([2.5, 2.5, 2.5]), env.root_states[0, 0:3])

    # set clock inputs
    env.commands[:, 4] = step_frequency
    env.commands[:, 5:8] = gait
    phase_0 = to_torch(obs[:, 17])
    env.gait_indices = to_torch(obs[:, 17])
    env.clock_inputs = to_torch(obs[:, 13:17])

    obs = np.concatenate([
                                to_np([[0.,0.]]),
                                to_np(env.root_states[:,2:3]), to_np(env.root_states[:,3:7]),
                                to_np(env.root_states[:,7:10]), to_np(env.root_states[:,10:13]),
                                to_np(env.clock_inputs), to_np(env.gait_indices.unsqueeze(1)),
                                to_np(env.dof_pos[:,:12]), to_np(env.dof_vel[:, :12])], axis=-1)

    recorded_obs = [deepcopy(obs[:, None])]

    while t < 56:
        if not replay:
            obs = dataset.normalizer.normalize(obs, 'observations')
            conditions = {0: to_torch(obs, device=device)}

            # state trajectory sampling
            samples = trainer.ema_model.conditional_sample(conditions, returns)

            # action sampling
            obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)
            obs_comb = obs_comb.reshape(-1, 2 * observation_dim)
            action = trainer.ema_model.inv_model(obs_comb)

            action = to_np(action)
            action = dataset.normalizer.unnormalize(action, 'actions')
            action = to_torch(action)

            #trajectory recording
            samples = to_np(samples)
            if t == 0:
                normed_observations = samples[:, :, :]
                observations = dataset.normalizer.unnormalize(normed_observations, 'observations')
                savepath = os.path.join('images', 'sample-planned.png')
                renderer.composite2(savepath, observations, str(t))

        elif replay:
            action = to_np(x[:, t, :action_dim])
            action = dataset.normalizer.unnormalize(action, 'actions')
            action = to_torch(action)

        obs_list = []
        with torch.no_grad():
            env.commands[:, 4] = step_frequency
            env.commands[:, 5:8] = gait
            if t==0: env.gait_indices = phase_0

            this_obs, this_reward, this_done, _ = env.step(action)
            env.set_camera(env.root_states[0, 0:3] + to_torch([2.5, 2.5, 2.5]), env.root_states[0, 0:3])

        # env_list[0].commands[:, 4] = step_frequency
        # env_list[0].commands[:, 5:8] = gait

        this_obs = np.concatenate([ to_np([[0.,0.]]), to_np(env.root_states[:,2:3]), to_np(env.root_states[:,3:7]),
                                        to_np(env.root_states[:,7:10]), to_np(env.root_states[:,10:13]),
                                        to_np(env.clock_inputs), to_np(env.gait_indices.unsqueeze(1)),
                                        to_np(env.dof_pos[:,:12]), to_np(env.dof_vel[:,:12])], axis=-1)

        obs_list.append(this_obs)
        if this_done:
            if dones[0] == 1:
                pass
            else:
                dones[0] = 1
                episode_rewards[0] += env.base_lin_vel[0,0].cpu() / 15.
                logger.print(f"Episode ({0}): {episode_rewards[0]}", color='green')
        else:
            if dones[0] == 1:
                pass
            else:
                episode_rewards[0] += env.base_lin_vel[0,0].cpu() / 15.

        obs = np.concatenate(obs_list, axis=0)
        recorded_obs.append(deepcopy(obs[:, None]))
        t += 1


    print("###################### end of simulation ######################")
    print(len(recorded_obs))

    clock_true = np.zeros(151)
    clock_inputs = np.zeros(151)
    for i, obs in enumerate(recorded_obs):
        clock_inputs[i] = obs[0][0][13]
    for i in range(56):
        clock_true[i] = targ_unnormed[0][i][13]
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(12, 5))

    axs[0].plot(np.linspace(0, 151 * env.dt, 151), clock_true, linestyle="-", label="Measured")
    axs[0].set_title("True Clock inputs")

    axs[1].plot(np.linspace(0, 151 * env.dt, 151), clock_inputs, linestyle="-", label="Measured")
    axs[1].set_title("Clock inputs")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Phase")

    plt.tight_layout()
    plt.show()

    print("코드 실행 시간:", sampling_time, "초")

    recorded_obs = np.concatenate(recorded_obs, axis=1)
    savepath = os.path.join('images', f'sample-executed.png')
    renderer.composite2(savepath, recorded_obs, 'rollout')
    episode_rewards = np.array(episode_rewards)

    logger.print(f"average_ep_reward: {np.mean(episode_rewards)}, std_ep_reward: {np.std(episode_rewards)}",
                 color='green')
    logger.log_metrics_summary(
        {'average_ep_reward': np.mean(episode_rewards), 'std_ep_reward': np.std(episode_rewards)})