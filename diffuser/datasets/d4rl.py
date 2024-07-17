import os
import collections
import numpy as np
import gym
import pdb

import pickle as pkl

from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)

@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

with suppress_output():
    ## d4rl prints out a variety of warnings
    import d4rl

#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#

def load_environment(name):
    if type(name) != str:
        ## name is already an environment
        return name
    with suppress_output():
        wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    return env

def get_dataset(env):
    dataset = env.get_dataset()
    return dataset

def sequence_dataset(env, preprocess_fn):
    """
    Returns an iterator through trajectories.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """

    file_path = os.path.expanduser('~/Desktop/data.pkl')
    with open(file_path, 'rb') as f:
        loaded_data = pkl.load(f)
        dataset = loaded_data

    file_paths = ['~/Desktop/dataset/0627_gait_joy_faster/trot/data.pkl',
                  '~/Desktop/dataset/0627_gait_joy_faster/bound/data.pkl',
                  '~/Desktop/dataset/0627_gait_joy_faster/pace/data.pkl',
                  '~/Desktop/dataset/0627_gait_joy_faster/pronk/data.pkl']
    dataset = {}
    keys = ['actions', 'observations', 'rewards', 'terminals', 'timeouts']

    # for file_path in file_paths:
    #     path = os.path.expanduser(file_path)
    #     with open(path, 'rb') as f:
    #         loaded_data = pkl.load(f)
    #     for key in keys:
    #         if key in dataset:
    #             dataset[key] = np.concatenate([dataset[key], loaded_data[key]], axis=0)
    #         else:
    #             dataset[key] = loaded_data[key]
    #
    # dataset = preprocess_fn(dataset)
    # generate_pos = True
    # include_clock = False
    #
    # seperate_gait = True
    #
    # if not include_clock:
    #     dataset['observations'] = np.concatenate([dataset['observations'][:,0:13], dataset['observations'][:,18:]], axis=-1)
    #     print(dataset['observations'].shape)
    #
    # if not generate_pos:
    #     dataset['observations'] = dataset['observations'][:,2:]
    #
    # if not seperate_gait:
    #     dataset['rewards'][:,0] = -1


    file_path = os.path.expanduser('~/Desktop/backflip.pkl')
    with open(file_path, 'rb') as f:
        loaded_data = pkl.load(f)
    this_obs = np.concatenate([loaded_data['gc'][:,:,0:3], loaded_data['gc'][:,:,4:7], loaded_data['gc'][:,:,3:4],
                               loaded_data['gv'][:,:,0:6], loaded_data['gc'][:,:,7:], loaded_data['gv'][:,:,6:]], axis=-1)
    this_acts = loaded_data['torque']
    # this_acts = 0.05 * (loaded_data['torque'] + 0.5 * loaded_data['gv'][:,:,6:]) + loaded_data['gc'][:,:,7:]
    # this_acts -= np.array([0.1000, 0.8000, -1.5000, -0.1000, 0.8000, -1.5000, 0.1000, 1.0000, -1.5000, -0.1000, 1.0000, -1.5000])
    this_obs = this_obs.reshape(-1, this_obs.shape[-1])
    this_acts = this_acts.reshape(-1, this_acts.shape[-1])
    this_rewards = np.zeros((4000*76,4))
    this_rewards[:,:] = np.array([4, 0, 0, 0])
    this_terminals = np.zeros((4000 * 76, 1))
    this_terminals = this_terminals.astype(bool)
    this_terminals = this_terminals.reshape(-1)
    this_timeouts = loaded_data['timeouts'].astype(bool)
    this_timeouts = this_timeouts.reshape(-1)
    this_data = [this_acts, this_obs, this_rewards, this_terminals, this_timeouts]

    for i, key in enumerate(keys):
        if key in dataset:
            dataset[key] = np.concatenate([dataset[key], this_data[i]], axis=0)
        else:
            dataset[key] = this_data[i]

    file_path = os.path.expanduser('~/Desktop/sideflip.pkl')
    with open(file_path, 'rb') as f:
        loaded_data = pkl.load(f)
    this_obs = np.concatenate([loaded_data['gc'][:,:,0:3], loaded_data['gc'][:,:,4:7], loaded_data['gc'][:,:,3:4],
                               loaded_data['gv'][:,:,0:6], loaded_data['gc'][:,:,7:], loaded_data['gv'][:,:,6:]], axis=-1)
    this_acts = loaded_data['torque']
    # this_acts = 0.05 * (loaded_data['torque'] + 0.5 * loaded_data['gv'][:,:,6:]) + loaded_data['gc'][:,:,7:]
    # this_acts -= np.array([0.1000, 0.8000, -1.5000, -0.1000, 0.8000, -1.5000, 0.1000, 1.0000, -1.5000, -0.1000, 1.0000, -1.5000])
    this_obs = this_obs.reshape(-1, this_obs.shape[-1])
    this_acts = this_acts.reshape(-1, this_acts.shape[-1])
    this_rewards = np.zeros((4000*76,4))
    this_rewards[:,:] = np.array([5, 0, 0, 0])
    this_terminals = np.zeros((4000 * 76, 1))
    this_terminals = this_terminals.astype(bool)
    this_terminals = this_terminals.reshape(-1)
    this_timeouts = loaded_data['timeouts'].astype(bool)
    this_timeouts = this_timeouts.reshape(-1)
    this_data = [this_acts, this_obs, this_rewards, this_terminals, this_timeouts]

    for i, key in enumerate(keys):
        if key in dataset:
            dataset[key] = np.concatenate([dataset[key], this_data[i]], axis=0)
        else:
            dataset[key] = this_data[i]

    file_path = os.path.expanduser('~/Desktop/yawspin.pkl')
    with open(file_path, 'rb') as f:
        loaded_data = pkl.load(f)
    this_obs = np.concatenate([loaded_data['gc'][:,:,0:3], loaded_data['gc'][:,:,4:7], loaded_data['gc'][:,:,3:4],
                               loaded_data['gv'][:,:,0:6], loaded_data['gc'][:,:,7:], loaded_data['gv'][:,:,6:]], axis=-1)
    this_acts = loaded_data['torque']
    # this_acts = 0.05 * (loaded_data['torque'] + 0.5 * loaded_data['gv'][:,:,6:]) + loaded_data['gc'][:,:,7:]
    # this_acts -= np.array([0.1000, 0.8000, -1.5000, -0.1000, 0.8000, -1.5000, 0.1000, 1.0000, -1.5000, -0.1000, 1.0000, -1.5000])
    this_obs = this_obs.reshape(-1, this_obs.shape[-1])
    this_acts = this_acts.reshape(-1, this_acts.shape[-1])
    this_rewards = np.zeros((4000*76,4))
    this_rewards[:,:] = np.array([6, 0, 0, 0])
    this_terminals = np.zeros((4000 * 76, 1))
    this_terminals = this_terminals.astype(bool)
    this_terminals = this_terminals.reshape(-1)
    this_timeouts = loaded_data['timeouts'].astype(bool)
    this_timeouts = this_timeouts.reshape(-1)
    this_data = [this_acts, this_obs, this_rewards, this_terminals, this_timeouts]

    for i, key in enumerate(keys):
        if key in dataset:
            dataset[key] = np.concatenate([dataset[key], this_data[i]], axis=0)
        else:
            dataset[key] = this_data[i]


    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = 'timeouts' in dataset

    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = bool(dataset['timeouts'][i])
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)

        for k in dataset:
            if 'metadata' in k: continue
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            if done_bool:
                print("here")
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1
        # if episode_step % 1000 == 0: print(episode_step)

