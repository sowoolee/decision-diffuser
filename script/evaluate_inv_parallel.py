import isaacgym
assert isaacgym

import diffuser.utils as utils
from ml_logger import logger
import torch
from copy import deepcopy
import numpy as np
import os
import gym
from config.locomotion_config import Config
from diffuser.utils.arrays import to_torch, to_np, to_device
from diffuser.datasets.d4rl import suppress_output

import glob
import pickle as pkl
from go1_gym.envs import *
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv
import time

from tqdm import tqdm

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

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "actuator_net"

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
    returns = to_device(torch.Tensor([[1,0,0] for i in range(num_eval)]), device)

    trot_returns = to_device(torch.Tensor([[1,0,0] for i in range(num_eval)]), device)
    bound_returns = to_device(torch.Tensor([[0,1,0] for i in range(num_eval)]), device)
    pace_returns = to_device(torch.Tensor([[0,0,1] for i in range(num_eval)]), device)
    #####################################################################

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
    # obs_list = [env.reset().detach().cpu() for env in env_list]
    # obs = np.concatenate(obs_list, axis=0)
    # obs = np.concatenate([obs[:, :3], obs[:, 66:70], obs[:, 18:30] + default_pos, obs[:, 30:42]], axis=-1)
    ##########################################################

    ################ can evalutate just one env ##################
    ################ play env obdim 42 ver #######################
    for env in env_list:
        obs_list = []
        obs_list.append(env.reset().detach().cpu())
        obs = np.concatenate(obs_list, axis=0)
        obs = np.concatenate([ to_np(env.root_states[:,2:3]), to_np(env.root_states[:,3:7]),
                                    to_np(env.root_states[:,7:10]), to_np(env.base_ang_vel[:,:]),
                                    obs[:, :3], obs[:, 66:70], obs[:, 18:30] + default_pos, obs[:, 30:42]], axis=-1)
    ##############################################################

    recorded_obs = [deepcopy(obs[:, None])]
    warm_sample = None

    # while sum(dones) <  num_eval:
    while t < 80:
        obs = dataset.normalizer.normalize(obs, 'observations')
        conditions = {0: to_torch(obs, device=device)}

        if t == 0 :
            samples = trainer.ema_model.conditional_sample(conditions, returns=trot_returns)
            # torch.save(samples, 'initial_sample_pace.pt')
            samples = torch.load('initial_sample.pt')
            warm_sample = samples
        ############################## just one gait ###########################################
        # else :
        #     samples = trainer.ema_model.conditional_warm_sample(conditions, warm_sample, steps=10, returns=returns)
        #     warm_sample = samples
        ########################################################################################
        ############################# mixing gait ##############################################
        else :
            if t < 500:
                start_time = time.time()
                samples = trainer.ema_model.conditional_warm_sample(conditions, warm_sample, steps=9, returns=trot_returns)
                end_time = time.time()
                execution_time = end_time - start_time
                sampling_time += execution_time
                warm_sample = samples

            elif t == 170:
                samples = trainer.ema_model.conditional_sample(conditions, returns=bound_returns)
                warm_sample = samples
            else:
                samples = trainer.ema_model.conditional_warm_sample(conditions, warm_sample, steps=9, returns=bound_returns)
                warm_sample = samples
        ########################################################################################

        obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)
        obs_comb = obs_comb.reshape(-1, 2 * observation_dim)
        action = trainer.ema_model.inv_model(obs_comb)

        samples = to_np(samples)
        action = to_np(action)

        action = dataset.normalizer.unnormalize(action, 'actions')
        action = to_torch(action[None])

        # if t == 0 :
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
            ######################### play env ver ###########################################
            # this_obs = np.concatenate([this_obs[:, :3], this_obs[:, 66:70],
            #                            this_obs[:, 18:30] + default_pos, this_obs[:, 30:42]], axis=-1)
            ##################################################################################
            ######################### play env obdim 42 ver ##################################
            this_obs = np.concatenate([ to_np(env_list[i].root_states[:,2:3]), to_np(env_list[i].root_states[:,3:7]),
                                            to_np(env_list[i].root_states[:,7:10]), to_np(env_list[i].base_ang_vel[:,:]),
                                            this_obs[:, :3], this_obs[:, 66:70],
                                            this_obs[:, 18:30] + default_pos, this_obs[:, 30:42]], axis=-1)
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


    print("코드 실행 시간:", sampling_time, "초")
    print("average frequency:", 80/sampling_time, "Hz")


    recorded_obs = np.concatenate(recorded_obs, axis=1)
    savepath = os.path.join('images', f'sample-executed.png')
    # renderer.composite(savepath, recorded_obs)
    episode_rewards = np.array(episode_rewards)

    logger.print(f"average_ep_reward: {np.mean(episode_rewards)}, std_ep_reward: {np.std(episode_rewards)}",
                 color='green')
    logger.log_metrics_summary(
        {'average_ep_reward': np.mean(episode_rewards), 'std_ep_reward': np.std(episode_rewards)})
