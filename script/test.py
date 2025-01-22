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
from diffuser.models.helpers import apply_conditioning

from isaacgym.torch_utils import quat_rotate_inverse

import onnx
import onnxruntime as ort

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

    # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete flat, gap, smooth plain, ]
    Cfg.terrain.terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1]

    # discrete_obstacles_height = 0.05 + difficulty * (cfg.max_platform_height - 0.05)
    Cfg.terrain.slope_treshold = 0.
    Cfg.terrain.max_platform_height = 0.
    Cfg.terrain.difficulty_scale = 1.
    print("default slope : ", Cfg.terrain.slope_treshold)
    print("difficulty : ", Cfg.terrain.difficulty_scale)
    print("platform height : ", Cfg.terrain.max_platform_height)

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


def import_diffuser(path):
    import diffuser.utils as utils
    from ml_logger import logger, RUN
    from config.locomotion_config import Config

    # logger.remove('*.pkl')
    # logger.remove("traceback.err")
    # logger.log_params(Config=vars(Config), RUN=vars(RUN))

    Config.device = 'cuda:0'

    basepath = '/home/hubolab/workspace/DD/weights/diffuser/'
    # basepath = '/home/hubolab/workspace/DD/weights/diffuser/4_gaits/mamba/checkpoint'
    # loadpath = '/home/hubolab/workspace/DD/weights/diffuser3/ADD/mamba/checkpoint'
    loadpath = os.path.join(basepath, path, 'checkpoint', 'state_50000.pt')
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
    # renderer = render_config()
    renderer = None

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
    renderer = trainer.renderer

    # trainer.record_samples()

    # load environment
    num_envs = env.num_envs

    # y conditioning
    gait_num = 1
    v_x = 1.5

    # start testing
    t = 0
    env.reset()
    total_steps = 100
    state_traj = []
    inference_time = 0

    measured_x_vels = np.zeros(total_steps)
    measured_y_vels = np.zeros(total_steps)
    planned_x_vels = np.zeros(total_steps)
    planned_y_vels = np.zeros(total_steps)
    target_x_vels = np.ones(total_steps) * v_x

    while t < total_steps:
        #if t < total_steps * 0.4:
        #    gait_num = 3
        #else:
        #    gait_num = 0
        returns = to_device(torch.Tensor([[gait_num, v_x, 0,0] for i in range(num_envs)]), device)

        obs = np.concatenate([
            to_np([[0.,0.]]),
            to_np(env.root_states[:,2:3]), to_np(env.root_states[:,3:7]),
            to_np(env.root_states[:,7:10]), to_np(env.root_states[:,10:13]),
            to_np(env.dof_pos[:,:12]), to_np(env.dof_vel[:, :12])], axis=-1)

        s_t = np.concatenate([to_np(env.root_states[:,0:2]), obs[:,2:]], axis=-1)
        state_traj.append(s_t)

        # action sampling
        obs = dataset.normalizer.normalize(obs, 'observations')
        obs = np.concatenate([to_np([[0.3,0.3]]), obs[:,2:]], axis=-1)

        conditions = {0: to_torch(obs, device=device)}

        # state trajectory sampling
        start = time.time()
        samples = trainer.ema_model.conditional_sample(conditions, returns)
        end = time.time()
        inference_time += (end - start)
        obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)

        if t==30:
            planned_linvel = to_np(
                quat_rotate_inverse(to_torch(dataset.normalizer.unnormalize(to_np(samples), 'observations')[0,:,3:7]), to_torch(dataset.normalizer.unnormalize(to_np(samples), 'observations')[0,:,7:10]))
            )
            for i in range(planned_linvel.shape[0]):
                planned_x_vels[i] = planned_linvel[i][0]
                planned_y_vels[i] = planned_linvel[i][1]
        # quat_rotate_inverse(to_torch(dataset.normalizer.unnormalize(to_np(samples), 'observations')[0,:,3:7]), to_torch(dataset.normalizer.unnormalize(to_np(samples), 'observations')[0,:,7:10]))

        with torch.no_grad():
            action = trainer.ema_model.inv_model(obs_comb)
            env.step(action)
            env.set_camera(env.root_states[0, 0:3] + to_torch([2.5, 2.5, 2.5]), env.root_states[0, 0:3])

        measured_x_vels[t] = env.base_lin_vel[0, 0]
        measured_y_vels[t] = env.base_lin_vel[0, 1]

        print("Environment timestep: {}".format(t))

        t += 1

    print('evaluation ended')

    target_vel = np.array([v_x, 0])
    planned_xy = planned_linvel[:,:2]
    measured_xy = np.stack([measured_x_vels, measured_y_vels], axis=1)
    diff = measured_xy - target_vel # planned_xy - target_vel
    velocity_norms = np.linalg.norm(diff, axis=1)  # (56,)
    # print("Velocity differences (norm):", velocity_norms)
    print("Average Velocity Tracking RMS Error: ", np.mean(velocity_norms))

    print("Average Inference Time: ", inference_time / total_steps, "s")

    # play recorded trajectory
    recorded_traj = np.stack(state_traj, axis=1)
    renderer.composite3('test', recorded_traj, 'trot')

    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(12, 5))
    axs[0].plot(np.linspace(0, total_steps * 0.02, total_steps), measured_x_vels, color='black', linestyle="-", label="Measured_x")
    axs[0].plot(np.linspace(0, total_steps * 0.02, total_steps), measured_y_vels, color='black', linestyle="-", label="Measured_y")
    axs[0].plot(np.linspace(0, total_steps * 0.02, total_steps), target_x_vels, color='black', linestyle="--", label="Desired")
    axs[0].legend()
    axs[0].set_title("Forward Linear Velocity")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Velocity (m/s)")

    axs[1].plot(np.linspace(0, total_steps * 0.02, total_steps), planned_x_vels, color='black', linestyle="-", label="Measured_x")
    axs[1].plot(np.linspace(0, total_steps * 0.02, total_steps), planned_y_vels, color='black', linestyle="-", label="Measured_y")
    axs[1].plot(np.linspace(0, total_steps * 0.02, total_steps), target_x_vels, color='black', linestyle="--", label="Desired")
    axs[1].legend()
    axs[1].set_title("Planned Forward Linear Velocity")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Velocity (m/s)")

    plt.tight_layout()
    plt.show()


def test_add():
    label = "gait-conditioned-agility/pretrain-v0/train"
    env = load_env(label, headless=False)

    # import diffusion model
    trainer = import_diffuser('4gait_history/unet')
    dataset = trainer.dataset
    device = trainer.device
    renderer = trainer.renderer

    # trainer.record_samples_acc()

    # load environment
    num_envs = env.num_envs

    # y conditioning
    gait_num = 1
    v_x = 1.5

    # start testing
    t = 0
    env.reset()
    total_steps = 500
    state_traj = []
    inference_time = 0

    measured_x_vels = np.zeros(total_steps)
    measured_y_vels = np.zeros(total_steps)
    planned_x_vels = np.zeros(total_steps)
    planned_y_vels = np.zeros(total_steps)
    target_x_vels = np.ones(total_steps) * v_x

    from collections import deque
    history_buffer = deque(maxlen=3)

    while t < total_steps:
        if t < total_steps * 0.4:
           gait_num = 0
        else:
           gait_num = 1
        returns = to_device(torch.Tensor([[gait_num, v_x, 0,0] for i in range(num_envs)]), device)

        obs = np.concatenate([
            to_np([[0.,0.]]),
            to_np(env.root_states[:,2:3]), to_np(env.root_states[:,3:7]),
            to_np(env.root_states[:,7:10]), to_np(env.root_states[:,10:13]),
            to_np(env.dof_pos[:,:12]), to_np(env.dof_vel[:, :12])], axis=-1)

        s_t = np.concatenate([to_np(env.root_states[:,0:2]), obs[:,2:]], axis=-1)
        state_traj.append(s_t)

        # action sampling
        obs = dataset.normalizer.normalize(obs, 'observations')
        obs = np.concatenate([to_np([[0.3,0.3]]), obs[:,2:]], axis=-1)

        current_history = [obs[:,2:]]

        conditions = {0: to_torch(obs, device=device)}

        # state trajectory sampling
        start = time.time()
        # samples = trainer.ema_model.conditional_sample_acc(conditions, returns)

        x = torch.randn(1,56,37).to(device)
        x = apply_conditioning(x, conditions, 0)
        timestep = torch.full((1,), 99, device=x.device).long()

        if len(history_buffer) < 3:
            padded_history = [np.zeros((1,47), dtype=float) for _ in range(3 - len(history_buffer))]
            history = torch.tensor([np.concatenate(padded_history + list(history_buffer), axis=0)], device=device).float()
        else:
            history = torch.tensor([np.concatenate(history_buffer, axis=0)], device=device).float()

        samples = trainer.ema_model.model(x, conditions, timestep, returns, history)
        samples = apply_conditioning(samples, conditions, 0)

        end = time.time()
        inference_time += (end - start)
        obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)

        if t==30:
            planned_linvel = to_np(
                quat_rotate_inverse(to_torch(dataset.normalizer.unnormalize(to_np(samples), 'observations')[0,:,3:7]), to_torch(dataset.normalizer.unnormalize(to_np(samples), 'observations')[0,:,7:10]))
            )
            for i in range(planned_linvel.shape[0]):
                planned_x_vels[i] = planned_linvel[i][0]
                planned_y_vels[i] = planned_linvel[i][1]
        # quat_rotate_inverse(to_torch(dataset.normalizer.unnormalize(to_np(samples), 'observations')[0,:,3:7]), to_torch(dataset.normalizer.unnormalize(to_np(samples), 'observations')[0,:,7:10]))

        with torch.no_grad():
            action = trainer.ema_model.inv_model(obs_comb)
            current_history.append(to_np(action))
            env.step(action)
            env.set_camera(env.root_states[0, 0:3] + to_torch([2.5, 2.5, 2.5]), env.root_states[0, 0:3])

        measured_x_vels[t] = env.base_lin_vel[0, 0]
        measured_y_vels[t] = env.base_lin_vel[0, 1]

        current_history = np.concatenate(current_history, axis=-1)
        history_buffer.append(current_history)

        print("Environment timestep: {}".format(t))

        t += 1

    print('evaluation ended')

    target_vel = np.array([v_x, 0])
    planned_xy = planned_linvel[:,:2]
    measured_xy = np.stack([measured_x_vels, measured_y_vels], axis=1)
    diff = measured_xy - target_vel # planned_xy - target_vel
    velocity_norms = np.linalg.norm(diff, axis=1)  # (56,)
    # print("Velocity differences (norm):", velocity_norms)
    print("Average Velocity Tracking RMS Error: ", np.mean(velocity_norms))

    print("Average Inference Time: ", inference_time / total_steps, "s")

    # play recorded trajectory
    recorded_traj = np.stack(state_traj, axis=1)
    # renderer.composite3('test', recorded_traj, 'trot')

    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(12, 5))
    axs[0].plot(np.linspace(0, total_steps * 0.02, total_steps), measured_x_vels, color='black', linestyle="-", label="Measured_x")
    axs[0].plot(np.linspace(0, total_steps * 0.02, total_steps), measured_y_vels, color='black', linestyle="-", label="Measured_y")
    axs[0].plot(np.linspace(0, total_steps * 0.02, total_steps), target_x_vels, color='black', linestyle="--", label="Desired")
    axs[0].legend()
    axs[0].set_title("Forward Linear Velocity")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Velocity (m/s)")

    axs[1].plot(np.linspace(0, total_steps * 0.02, total_steps), planned_x_vels, color='black', linestyle="-", label="Measured_x")
    axs[1].plot(np.linspace(0, total_steps * 0.02, total_steps), planned_y_vels, color='black', linestyle="-", label="Measured_y")
    axs[1].plot(np.linspace(0, total_steps * 0.02, total_steps), target_x_vels, color='black', linestyle="--", label="Desired")
    axs[1].legend()
    axs[1].set_title("Planned Forward Linear Velocity")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Velocity (m/s)")

    plt.tight_layout()
    plt.show()


def export_diffuser_onnx():
    # import diffusion model
    trainer = import_diffuser('4_gaits/unet')
    device = trainer.device

    unet = trainer.ema_model.model.to('cpu')
    cond = {0: 0.1*torch.ones((1,37), device='cpu')}
    s0 = cond[0]
    returns = torch.tensor([[1,0,0,0]], dtype=torch.float, device='cpu')
    x = 0.5*torch.ones((1,56,37), device='cpu')
    t = torch.full((1,), 99, device='cpu', dtype=torch.long)

    #####################  exporting  ###########################
    torch.onnx.export(
        unet,
        (x, t, returns),
        "Unet.onnx",
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=['x', 'time', 'returns'],
        output_names=['output'],
    )
    #############################################################

    torch_out = unet(x, t, returns, use_dropout=False)

    onnx_model_path = "/home/hubolab/Desktop/Unet.onnx"

    model = onnx.load("MambaUnet.onnx")
    for input in model.graph.input:
        print(input.name)

    ort_session = ort.InferenceSession(onnx_model_path)

    ort_inputs = {
        'x': x.cpu().numpy(),
        'time': t.cpu().numpy(),
        'returns': returns.cpu().numpy(),
    }
    ort_outs = ort_session.run(None, ort_inputs)

    inf_t = 0
    for _ in range(80):
        start = time.time()
        ort_outs = ort_session.run(None, ort_inputs)
        end = time.time()
        inf_t += end - start
    print("onnx unet sampling time : {}".format(inf_t / 10))

    onnx_out_np = ort_outs[0]
    torch_out_np = torch_out.detach().cpu().numpy()

    difference_norm = (np.linalg.norm(onnx_out_np - torch_out_np))
    print(f"The norm of the difference between ONNX and PyTorch outputs is: {difference_norm}")
    return None

def export_dipo_onnx():
    # import diffusion model
    trainer = import_diffuser('4gait_history/unet')
    device = trainer.device

    class UnifiedModel(torch.nn.Module):
        def __init__(self, ema_model):
            super().__init__()
            self.model = ema_model.model.to('cpu')
            self.inv_model = ema_model.inv_model.to('cpu')
        def forward(self, x, cond, time, returns, history):
            samples = self.model(x, cond, time, returns, history)
            # samples.clamp_(-1., 1.)
            samples[:,0,:] = cond.clone()
            obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1).to('cpu')
            action = self.inv_model(obs_comb)
            return action[0]

    dipo = UnifiedModel(trainer.ema_model)

    obs = np.concatenate([
        to_np([[0.,0., 0.266, 0, 0, 0, 1]]),
        to_np([[0,0,0,0,0,0]]),
        to_np([[0.00, 0.7854, -1.5708, 0.00, 0.7854, -1.5708, 0.00, 0.7854, -1.5708, 0.00, 0.7854, -1.5708]]),
        to_np([[0,0,0,0,0,0,0,0,0,0,0,0]])], axis=-1)
    obs = trainer.dataset.normalizer.normalize(obs, 'observations')
    obs = np.concatenate([to_np([[0.,0.]]), obs[:,2:]], axis=-1)

    cond = 0.*torch.ones((1,37), device='cpu')
    cond = to_torch(obs, device='cpu')

    history = torch.zeros((1,3,47), device='cpu')
    history[:,2,:35] = torch.tensor(obs[:,2:], device='cpu')

    returns = torch.tensor([[0.0,0.0,0.0,0.0]], dtype=torch.float, device='cpu')
    x = 0.*torch.ones((1,56,37), device='cpu')
    # x = torch.randn((1,56,37), device='cpu')
    x[:,0,:] = cond.clone()
    t = torch.full((1,), 99.0, device='cpu', dtype=torch.float)

    #####################  exporting  #########################
    torch.onnx.export(
        dipo,
        (x, cond, t, returns, history),
        "DiPo.onnx",
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=['x', 'cond', 'time', 'returns', 'history'],
        output_names=['output'],
    )
    ###########################################################

    torch_out = dipo(x, cond, t, returns, history)

    onnx_model_path = "/home/hubolab/Desktop/DiPo_history.onnx"

    # onnx_model = onnx.load(onnx_model_path)
    # for input_tensor in onnx_model.graph.input:
    #     print(f"Input {input_tensor.name}: {input_tensor.type.tensor_type.elem_type}")

    # model = onnx.load("DiPo.onnx")
    # for input in model.graph.input:
    #     print(input.name)

    ort_session = ort.InferenceSession(onnx_model_path)

    ort_inputs = {
        'x': x.cpu().numpy(),
        'cond': cond.cpu().numpy(),
        'time': t.cpu().numpy(),
        'returns': returns.cpu().numpy(),
        'history': history.cpu().numpy()
    }
    ort_outs = ort_session.run(None, ort_inputs)

    inf_t = 0
    for _ in range(80):
        start = time.time()
        ort_outs = ort_session.run(None, ort_inputs)
        end = time.time()
        inf_t += end - start
    print("onnx unet sampling time : {}".format(inf_t / 10))

    onnx_out_np = ort_outs[0]
    torch_out_np = torch_out.detach().cpu().numpy()

    difference_norm = (np.linalg.norm(onnx_out_np - torch_out_np))
    print(f"The norm of the difference between ONNX and PyTorch outputs is: {difference_norm}")
    return None

def test_onnx():
    label = "gait-conditioned-agility/pretrain-v0/train"
    env = load_env(label, headless=False)

    # import diffusion model
    trainer = import_diffuser('4_gaits/unet')
    dataset = trainer.dataset
    device = trainer.device

    onnx_model_path = "/home/hubolab/Desktop/Unet.onnx"
    ort_session = ort.InferenceSession(onnx_model_path)

    # load environment
    num_envs = env.num_envs

    # y conditioning
    gait_num = 1
    v_x = 1.5

    # start testing
    t = 0
    env.reset()
    total_steps = 200
    state_traj = []
    inference_time = 0

    measured_x_vels = np.zeros(total_steps)
    measured_y_vels = np.zeros(total_steps)
    planned_x_vels = np.zeros(total_steps)
    planned_y_vels = np.zeros(total_steps)
    target_x_vels = np.ones(total_steps) * v_x

    while t < total_steps:
        returns = to_device(torch.Tensor([[gait_num, v_x, 0,0] for i in range(num_envs)]), device)

        obs = np.concatenate([
            to_np([[0.,0.]]),
            to_np(env.root_states[:,2:3]), to_np(env.root_states[:,3:7]),
            to_np(env.root_states[:,7:10]), to_np(env.root_states[:,10:13]),
            to_np(env.dof_pos[:,:12]), to_np(env.dof_vel[:, :12])], axis=-1)

        s_t = np.concatenate([to_np(env.root_states[:,0:2]), obs[:,2:]], axis=-1)
        state_traj.append(s_t)

        # action sampling
        obs = dataset.normalizer.normalize(obs, 'observations')
        obs = np.concatenate([to_np([[0.3,0.3]]), obs[:,2:]], axis=-1)

        conditions = {0: to_torch(obs, device=device)}

        # state trajectory sampling
        start = time.time()
        # samples = trainer.ema_model.conditional_sample_acc(conditions, returns)

        x = torch.randn(1,56,37).to(device)
        x = apply_conditioning(x, conditions, 0)
        timestep = torch.full((1,), 99., device=x.device).long()

        ort_inputs = {
            'x': x.cpu().numpy(),
            'time': timestep.cpu().numpy(),
            'returns': returns.cpu().numpy(),
        }
        samples = ort_session.run(None, ort_inputs)
        samples = to_torch(samples[0], device=device)
        samples = apply_conditioning(samples, conditions, 0)

        end = time.time()
        inference_time += (end - start)
        obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)

        if t==30:
            planned_linvel = to_np(
                quat_rotate_inverse(to_torch(dataset.normalizer.unnormalize(to_np(samples), 'observations')[0,:,3:7]), to_torch(dataset.normalizer.unnormalize(to_np(samples), 'observations')[0,:,7:10]))
            )
            for i in range(planned_linvel.shape[0]):
                planned_x_vels[i] = planned_linvel[i][0]
                planned_y_vels[i] = planned_linvel[i][1]
        # quat_rotate_inverse(to_torch(dataset.normalizer.unnormalize(to_np(samples), 'observations')[0,:,3:7]), to_torch(dataset.normalizer.unnormalize(to_np(samples), 'observations')[0,:,7:10]))

        with torch.no_grad():
            action = trainer.ema_model.inv_model(obs_comb)
            env.step(action)
            env.set_camera(env.root_states[0, 0:3] + to_torch([2.5, 2.5, 2.5]), env.root_states[0, 0:3])

        measured_x_vels[t] = env.base_lin_vel[0, 0]
        measured_y_vels[t] = env.base_lin_vel[0, 1]

        print("Environment timestep: {}".format(t))

        t += 1

    print('evaluation ended')

    target_vel = np.array([v_x, 0])
    planned_xy = planned_linvel[:,:2]
    measured_xy = np.stack([measured_x_vels, measured_y_vels], axis=1)
    diff = measured_xy - target_vel # planned_xy - target_vel
    velocity_norms = np.linalg.norm(diff, axis=1)  # (56,)
    # print("Velocity differences (norm):", velocity_norms)
    print("Average Velocity Tracking RMS Error: ", np.mean(velocity_norms))
    print("Average Inference Time: ", inference_time / total_steps, "s")


    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(12, 5))
    axs[0].plot(np.linspace(0, total_steps * 0.02, total_steps), measured_x_vels, color='black', linestyle="-", label="Measured_x")
    axs[0].plot(np.linspace(0, total_steps * 0.02, total_steps), measured_y_vels, color='black', linestyle="-", label="Measured_y")
    axs[0].plot(np.linspace(0, total_steps * 0.02, total_steps), target_x_vels, color='black', linestyle="--", label="Desired")
    axs[0].legend()
    axs[0].set_title("Forward Linear Velocity")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Velocity (m/s)")

    axs[1].plot(np.linspace(0, total_steps * 0.02, total_steps), planned_x_vels, color='black', linestyle="-", label="Measured_x")
    axs[1].plot(np.linspace(0, total_steps * 0.02, total_steps), planned_y_vels, color='black', linestyle="-", label="Measured_y")
    axs[1].plot(np.linspace(0, total_steps * 0.02, total_steps), target_x_vels, color='black', linestyle="--", label="Desired")
    axs[1].legend()
    axs[1].set_title("Planned Forward Linear Velocity")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Velocity (m/s)")

    plt.tight_layout()
    plt.show()

def test_dipo():
    label = "gait-conditioned-agility/pretrain-v0/train"
    env = load_env(label, headless=False)

    # import diffusion model
    trainer = import_diffuser('4_gaits/unet')
    # trainer = import_diffuser('4gait_rand/unet')
    dataset = trainer.dataset
    device = trainer.device

    onnx_model_path = "/home/hubolab/Desktop/DiPo.onnx"
    # onnx_model_path = "/home/hubolab/workspace/DD/script/DiPo_rand.onnx"
    ort_session = ort.InferenceSession(onnx_model_path)

    # load environment
    num_envs = env.num_envs

    # y conditioning
    gait_num = 1
    v_x = 1.5

    # start testing
    t = 0
    env.reset()
    total_steps = 200
    state_traj = []
    inference_time = 0

    measured_x_vels = np.zeros(total_steps)
    measured_y_vels = np.zeros(total_steps)
    planned_x_vels = np.zeros(total_steps)
    planned_y_vels = np.zeros(total_steps)
    target_x_vels = np.ones(total_steps) * v_x

    while t < total_steps:
        returns = to_device(torch.Tensor([[gait_num, v_x, 0,0] for i in range(num_envs)]), device)

        obs = np.concatenate([
            to_np([[0.,0.]]),
            to_np(env.root_states[:,2:3]), to_np(env.root_states[:,3:7]),
            to_np(env.root_states[:,7:10]), to_np(env.root_states[:,10:13]),
            to_np(env.dof_pos[:,:12]), to_np(env.dof_vel[:, :12])], axis=-1)

        s_t = np.concatenate([to_np(env.root_states[:,0:2]), obs[:,2:]], axis=-1)
        state_traj.append(s_t)

        # action sampling
        obs = dataset.normalizer.normalize(obs, 'observations')
        obs = np.concatenate([to_np([[0.,0.]]), obs[:,2:]], axis=-1)

        conditions = {0: to_torch(obs, device=device)}

        # state trajectory sampling
        start = time.time()
        # samples = trainer.ema_model.conditional_sample_acc(conditions, returns)

        x = torch.randn(1,56,37).to(device)
        x = apply_conditioning(x, conditions, 0)
        timestep = torch.full((1,), 99., device=x.device)

        ort_inputs = {
            'x': x.cpu().numpy(),
            'cond': conditions[0].cpu().numpy(),
            'time': timestep.cpu().numpy(),
            'returns': returns.cpu().numpy(),
        }
        action = ort_session.run(None, ort_inputs)
        action = to_torch(action, device=device)

        with torch.no_grad():
            env.step(action)
            env.set_camera(env.root_states[0, 0:3] + to_torch([2.5, 2.5, 2.5]), env.root_states[0, 0:3])

        measured_x_vels[t] = env.base_lin_vel[0, 0]
        measured_y_vels[t] = env.base_lin_vel[0, 1]

        print("Environment timestep: {}".format(t))

        t += 1

    print('evaluation ended')

    target_vel = np.array([v_x, 0])
    measured_xy = np.stack([measured_x_vels, measured_y_vels], axis=1)
    diff = measured_xy - target_vel # planned_xy - target_vel
    velocity_norms = np.linalg.norm(diff, axis=1)  # (56,)
    # print("Velocity differences (norm):", velocity_norms)
    print("Average Velocity Tracking RMS Error: ", np.mean(velocity_norms))
    print("Average Inference Time: ", inference_time / total_steps, "s")


    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(12, 5))
    axs[0].plot(np.linspace(0, total_steps * 0.02, total_steps), measured_x_vels, color='black', linestyle="-", label="Measured_x")
    axs[0].plot(np.linspace(0, total_steps * 0.02, total_steps), measured_y_vels, color='black', linestyle="-", label="Measured_y")
    axs[0].plot(np.linspace(0, total_steps * 0.02, total_steps), target_x_vels, color='black', linestyle="--", label="Desired")
    axs[0].legend()
    axs[0].set_title("Forward Linear Velocity")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Velocity (m/s)")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # test()
    # test_add()
    # test_onnx()
    export_dipo_onnx()
    # test_dipo()