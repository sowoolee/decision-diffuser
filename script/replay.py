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

torch.cuda.set_per_process_memory_fraction(fraction=0.4, device=0)
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
    Cfg.terrain.num_rows = 5
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = False

    Cfg.domain_rand.lag_timesteps = 1
    Cfg.domain_rand.randomize_lag_timesteps = False
    Cfg.control.control_type = "P"

    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=Cfg)
    env = HistoryWrapper(env)

    # load policy
    from ml_logger import logger
    from go1_gym_learn.ppo_cse.actor_critic import ActorCritic


    return env

def load_test_env(label, headless=False):
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
    Cfg.terrain.num_rows = 10
    Cfg.terrain.num_cols = 10
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = False

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

def save(headless=True):
    from ml_logger import logger

    from pathlib import Path
    from go1_gym import MINI_GYM_ROOT_DIR
    import glob
    import os

    label = "gait-conditioned-agility/pretrain-v0/train"

    env = load_env(label, headless=headless)

    num_envs = env.num_envs
    num_eval_steps = 250
    gaits = {"pronking": [0, 0, 0],
             "trotting": [0.5, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, 0.5]}

    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = -2.5, 0.0, 0.0
    body_height_cmd = 0.0
    step_frequency_cmd = 3.0
    gait = torch.tensor(gaits["pacing"]) # change gait here
    footswing_height_cmd = 0.08
    pitch_cmd = 0.0
    roll_cmd = 0.0
    stance_width_cmd = 0.25

    dataset = {}
    state_ = []
    action_ = []
    reward_ = []
    timeout_ = None
    episodes = 1

    set_init_state = True
    obs_replay = False

    for episode in range(episodes):
        x_vel_cmd = torch.tensor([random.uniform(-3.0, 3.0) for _ in range(num_envs)])
        y_vel_cmd = torch.tensor([random.uniform(-1.5, 1.5) for _ in range(num_envs)])
        yaw_vel_cmd = torch.tensor([random.uniform(-0.8, 0.8) for _ in range(num_envs)])

        x_vel_cmd = torch.tensor([1.5 for _ in range(num_envs)])
        y_vel_cmd = torch.tensor([0.0 for _ in range(num_envs)])
        yaw_vel_cmd = torch.tensor([0.0 for _ in range(num_envs)])

        # gait_index = [random.choice(range(len(gaits))) for _ in range(num_envs)]
        gait_index = [3 for _ in range(num_envs)] # just for one gait
        random_gaits = [list(gaits.values())[idx] for idx in gait_index]
        gait = torch.tensor(random_gaits)

        recorded_obs = []
        recorded_acts = []
        recorded_rewards = []

        done_envs = []

        file_path = os.path.expanduser('~/Desktop/data.pkl')
        with open(file_path, 'rb') as f:
            loaded_data = pkl.load(f)
            dataset = loaded_data

        obs = env.reset()
        init_xy = env.root_states[:,0:2].detach().cpu()

        if set_init_state:
            env_ids = torch.tensor([0]).to('cuda:0')
            obs = torch.Tensor([dataset['observations'][0]]).to('cuda:0')
            dof_pos = torch.Tensor(obs[:,17:29]).to('cuda:0')
            dof_vel = torch.Tensor(obs[:,29:41]).to('cuda:0')
            base_state = obs[:,0:13]

            env.set_idx_state(env_ids, dof_pos, dof_vel, base_state)

        for i in tqdm(range(num_eval_steps)):
            with torch.no_grad():
                # action from dataset
                states = torch.Tensor([dataset['observations'][i]])
                actions = torch.Tensor([dataset['actions'][i]])

            this_obs = torch.cat(
                [env.root_states[:, 0:2].detach().cpu(), env.root_states[:, 2:3].detach().cpu(),
                 env.root_states[:, 3:7].detach().cpu(),
                 env.root_states[:, 7:10].detach().cpu(), env.root_states[:, 10:13].detach().cpu(), # env.base_ang_vel[:, :].detach().cpu(),
                 env.clock_inputs.detach().cpu(),
                 env.dof_pos[:, :12].detach().cpu(), env.dof_vel[:, :12].detach().cpu()], dim=-1)  # (500,41)
            this_reward = torch.stack([torch.tensor(gait_index), x_vel_cmd, y_vel_cmd, yaw_vel_cmd], dim=-1)  # (500,4) size this_reward

            ''' compare s_t+1 '''

            recorded_obs.append(this_obs)
            recorded_acts.append(actions)
            recorded_rewards.append(this_reward)

            with torch.no_grad():
                env.commands[:, 0] = x_vel_cmd
                env.commands[:, 1] = y_vel_cmd
                env.commands[:, 2] = yaw_vel_cmd
                env.commands[:, 3] = body_height_cmd
                env.commands[:, 4] = step_frequency_cmd
                env.commands[:, 5:8] = gait
                env.commands[:, 8] = 0.5
                env.commands[:, 9] = footswing_height_cmd
                env.commands[:, 10] = pitch_cmd
                env.commands[:, 11] = roll_cmd
                env.commands[:, 12] = stance_width_cmd

                obs, rew, done, info = env.step(actions)

                if obs_replay:
                    env_ids = torch.tensor([0]).to('cuda:0')
                    obs = torch.Tensor([dataset['observations'][i+1]]).to('cuda:0')
                    dof_pos = torch.Tensor(obs[:, 17:29]).to('cuda:0')
                    dof_vel = torch.Tensor(obs[:, 29:41]).to('cuda:0')
                    base_state = torch.Tensor(obs[:, 0:13]).to('cuda:0')
                    env.set_idx_state(env_ids, dof_pos, dof_vel, base_state)

                # compare s_(i+1)
                err = states - this_obs
            # if True in done: print("here")

            done_indices = [index for index, value in enumerate(done) if value]
            for ind in done_indices:
                if ind not in done_envs:
                    done_envs.append(ind)

            if i > 150:
                for ind in range(num_envs):
                    if float(env.base_lin_vel[ind,0].detach().cpu()) < 1.5*0.8 and ind not in done_envs:
                        done_envs.append(ind)
                print(err)


        recorded_obs = torch.stack(recorded_obs, dim=1) # 250*(500,41) -> (500,250,41)
        recorded_acts = torch.stack(recorded_acts, dim=1)
        recorded_rewards = torch.stack(recorded_rewards, dim=1)

        sliced_obs = []
        sliced_acts = []
        sliced_rewards = []
        if len(done_envs) != 0:
            done_envs.sort()
            done_envs = [0] + done_envs + [env.num_envs]
            slice_indices = list(zip(done_envs[:-1], done_envs[1:]))
            for i, (a,b) in enumerate(slice_indices):
                if i != 0:
                    a += 1
                    slice_indices[i] = (a,b)
            for a,b in slice_indices:
                sliced_obs.append(recorded_obs[a:b,:,:])
                sliced_acts.append(recorded_acts[a:b,:,:])
                sliced_rewards.append(recorded_rewards[a:b,:,:])

        if len(sliced_obs) != 0:
            recorded_obs = torch.cat(sliced_obs, axis=0)
            recorded_acts = torch.cat(sliced_acts, axis=0)
            recorded_rewards = torch.cat(sliced_rewards, axis=0)

        recorded_obs = recorded_obs.view(-1, 41)
        recorded_acts = recorded_acts.view(-1, 12)
        recorded_rewards = recorded_rewards.view(-1, 4)

        state_.append(recorded_obs)
        action_.append(recorded_acts)
        reward_.append(recorded_rewards)

    state_ = torch.cat(state_, dim=0)
    action_ = torch.cat(action_, dim=0)
    reward_ = torch.cat(reward_, dim=0)

    state_ = state_.view(-1, 41)
    action_ = action_.view(-1, 12)
    reward_ = reward_.view(-1, 4)

    state_ = state_.detach().cpu().numpy()
    action_ = action_.detach().cpu().numpy()
    reward_ = reward_.detach().cpu().numpy()

    dataset['actions'] = action_
    dataset['observations'] = state_
    # dataset['rewards'] = np.array([0.9 for i in range(state_.shape[0])])
    # dataset['rewards'] = np.array([(gait_emb + dir_emb) for i in range(state_.shape[0])])
    dataset['rewards'] = reward_
    timeouts = [False for i in range(249)] + [True]
    dataset['terminals'] = np.array([False for i in range(state_.shape[0])])
    true_eps = int(state_.shape[0] / 250)
    dataset['timeouts'] = np.array(timeouts * true_eps)

    # file_path = os.path.expanduser('~/Desktop/data.pkl')
    # with open(file_path, 'wb') as f:
    #     pkl.dump(dataset, f)

    for key in dataset:
        print(f"Key: {key}, Shape: {dataset[key].shape}")


def import_diffuser():
    import diffuser.utils as utils
    from ml_logger import logger, RUN
    from config.locomotion_config import Config

    # logger.remove('*.pkl')
    # logger.remove("traceback.err")
    # logger.log_params(Config=vars(Config), RUN=vars(RUN))

    Config.device = 'cuda:1'

    loadpath = '/home/kdyun/workspace/decidiff/code/weights/diffuser/default_inv/predict_epsilon_200_1000000.0/dropout_0.25/hopper-medium-expert-v2/100/checkpoint'
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

    assert trainer.ema_model.condition_guidance_w == Config.condition_guidance_w

    return trainer


def test(headless=False):
    from ml_logger import logger

    from pathlib import Path
    from go1_gym import MINI_GYM_ROOT_DIR
    import glob
    import os

    # import diffusion model
    trainer = import_diffuser()
    dataset = trainer.dataset
    device = trainer.device
    renderer = trainer.renderer

    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim


    # load environment
    label = "gait-conditioned-agility/pretrain-v0/train"
    env = load_test_env(label, headless=headless)
    num_eval = env.num_envs

    num_eval_steps = 250
    gaits = {"pronking": [0, 0, 0],
             "trotting": [0.5, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, 0.5]}


    # y conditioning
    gait_idx = 1
    gait_indices = [gait_idx for _ in range(num_eval)]  # just for one gait
    random_gaits = [list(gaits.values())[idx] for idx in gait_indices]
    gait = torch.tensor(random_gaits)
    step_frequency = 3.0

    returns = to_device(torch.Tensor([[gait_idx,0.0,0.0,0.0] for i in range(num_eval)]), device)

    # evaluation setting
    replay = False
    random_start = True
    set_batch_state = False


    # bring train dataset
    dataloader = cycle(torch.utils.data.DataLoader(
            dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
    batch = next(dataloader)
    batch = batch_to_device(batch, device=device)

    x = batch[0]
    cond = batch[1]
    # if set_batch_state: returns = batch[2]

    state = to_np(x[:, :, action_dim:])
    state = dataset.normalizer.unnormalize(state, 'observations')
    true_action = to_np(x[:,:, :action_dim])

    obs_comb = np.concatenate([state[:,0,:], state[:,1,:]], axis=-1)
    a = trainer.ema_model.inv_model(to_torch(obs_comb, device=device))
    dataset.normalizer.unnormalize(to_np(a), 'actions') -dataset.normalizer.unnormalize(to_np(true_action[:,0,:]), 'actions')

    # testing start
    t = 0

    env.reset()
    obs = dataset.normalizer.unnormalize(to_np(cond[0]), 'observations')

    # set batch state
    if set_batch_state:
        env_ids = torch.tensor([0], dtype=torch.int32, device=device)
        dof_pos = to_torch(obs[:,18:30])
        dof_vel = to_torch(obs[:,30:42])
        base_state = to_torch([obs[0, 0:13]])
        base_state = to_torch(np.concatenate([to_np(env.root_states[:,0:2]), obs[:,2:13]], axis=-1))
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
    obs_shot = np.concatenate([to_np(env.root_states[:,0:2]), obs[:,2:]], axis=-1)

    recorded_obs = [deepcopy(obs_shot[:, None])]

    while t < 250:
        # action sampling
        if not replay:
            obs = dataset.normalizer.normalize(obs, 'observations')
            if random_start: obs = np.concatenate([to_np([[0.,0.]]), obs[:,2:]], axis=-1)
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
                scaled_xy = samples[:,:,0:2]
                observations = np.concatenate([scaled_xy, observations[:,:,2:]], axis=-1)
                savepath = None
                renderer.composite2(savepath, observations, 'plan')

        elif replay:
            action = to_np(x[:, t, :action_dim])
            action = dataset.normalizer.unnormalize(action, 'actions')
            action = to_torch(action)

        # environment step
        obs_list = []
        with torch.no_grad():
            env.commands[:, 4] = step_frequency
            env.commands[:, 5:8] = gait
            if t==0: env.gait_indices = phase_0

            env.step(action)
            env.set_camera(env.root_states[0, 0:3] + to_torch([2.5, 2.5, 2.5]), env.root_states[0, 0:3])

        this_obs = np.concatenate([ to_np([[0.,0.]]), to_np(env.root_states[:,2:3]), to_np(env.root_states[:,3:7]),
                                        to_np(env.root_states[:,7:10]), to_np(env.root_states[:,10:13]),
                                        to_np(env.clock_inputs), to_np(env.gait_indices.unsqueeze(1)),
                                        to_np(env.dof_pos[:,:12]), to_np(env.dof_vel[:,:12])], axis=-1)

        obs_list.append(this_obs)
        obs = np.concatenate(obs_list, axis=0)
        obs_shot = np.concatenate([to_np(env.root_states[:, 0:2]), obs[:, 2:]], axis=-1)
        recorded_obs.append(deepcopy(obs_shot[:, None]))

        # compare s_t+1 with dataset
        # err = state[:,t+1,:] - this_obs

        t += 1

    print('evaluation ended')

    recorded_obs = np.concatenate(recorded_obs, axis=1)
    savepath = None
    renderer.composite2(savepath, recorded_obs, 'trot_fwd_0.75_1M')

    # for i in tqdm(range(num_eval_steps)):
    #     with torch.no_grad():
    #         # action from dataset
    #         states = torch.Tensor([dataset['observations'][i]])
    #         actions = torch.Tensor([dataset['actions'][i]])
    #
    #     this_obs = torch.cat(
    #         [env.root_states[:, 0:2].detach().cpu(), env.root_states[:, 2:3].detach().cpu(),
    #          env.root_states[:, 3:7].detach().cpu(),
    #          env.root_states[:, 7:10].detach().cpu(), env.root_states[:, 10:13].detach().cpu(), # env.base_ang_vel[:, :].detach().cpu(),
    #          env.clock_inputs.detach().cpu(),
    #          env.dof_pos[:, :12].detach().cpu(), env.dof_vel[:, :12].detach().cpu()], dim=-1)  # (500,41)
    #
    #     ''' compare s_t+1 '''





if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    # save(headless=False)
    test()