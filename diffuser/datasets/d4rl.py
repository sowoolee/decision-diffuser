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

    if 'antmaze' in str(env).lower():
        ## the antmaze-v0 environments have a variety of bugs
        ## involving trajectory segmentation, so manually reset
        ## the terminal and timeout fields
        dataset = antmaze_fix_timeouts(dataset)
        dataset = antmaze_scale_rewards(dataset)
        get_max_delta(dataset)

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
    # dataset = get_dataset(env)

    # dataset = {}
    # dataset['actions'] = np.zeros((2500,12))
    # dataset['observations'] = np.zeros((2500,12))
    # dataset['rewards'] = np.ones((2500,))
    # terminals = [False for i in range(249)] + [True]
    # dataset['terminals'] = np.array(terminals * 10)
    # dataset['timeouts'] = np.array([False for i in range(2500)])

    ################### using only one dataset ##################
    # file_path = os.path.expanduser('~/Desktop/data.pkl')
    # with open(file_path, 'rb') as f:
    #     loaded_data = pkl.load(f)
    #     dataset = loaded_data
    #############################################################
    ################### using more datasets #######################
    tf_path = os.path.expanduser('~/Desktop/dataset/2500_obdim42_fblr/trot/forward/data.pkl')
    tb_path = os.path.expanduser('~/Desktop/dataset/2500_obdim42_fblr/trot/backward/data.pkl')
    tl_path = os.path.expanduser('~/Desktop/dataset/2500_obdim42_fblr/trot/left/data.pkl')
    tr_path = os.path.expanduser('~/Desktop/dataset/2500_obdim42_fblr/trot/right/data.pkl')

    bf_path = os.path.expanduser('~/Desktop/dataset/2500_obdim42_fblr/bound/forward/data.pkl')
    bb_path = os.path.expanduser('~/Desktop/dataset/2500_obdim42_fblr/bound/backward/data.pkl')
    bl_path = os.path.expanduser('~/Desktop/dataset/2500_obdim42_fblr/bound/left/data.pkl')
    br_path = os.path.expanduser('~/Desktop/dataset/2500_obdim42_fblr/bound/right/data.pkl')

    pf_path = os.path.expanduser('~/Desktop/dataset/2500_obdim42_fblr/pace/forward/data.pkl')
    pb_path = os.path.expanduser('~/Desktop/dataset/2500_obdim42_fblr/pace/backward/data.pkl')
    pl_path = os.path.expanduser('~/Desktop/dataset/2500_obdim42_fblr/pace/left/data.pkl')
    pr_path = os.path.expanduser('~/Desktop/dataset/2500_obdim42_fblr/pace/right/data.pkl')

    with open(tf_path, 'rb') as f:
        loaded_data = pkl.load(f)
        tf_dataset = loaded_data
    with open(tb_path, 'rb') as f:
        loaded_data = pkl.load(f)
        tb_dataset = loaded_data
    with open(tl_path, 'rb') as f:
        loaded_data = pkl.load(f)
        tl_dataset = loaded_data
    with open(tr_path, 'rb') as f:
        loaded_data = pkl.load(f)
        tr_dataset = loaded_data

    with open(bf_path, 'rb') as f:
        loaded_data = pkl.load(f)
        bf_dataset = loaded_data
    with open(bb_path, 'rb') as f:
        loaded_data = pkl.load(f)
        bb_dataset = loaded_data
    with open(bl_path, 'rb') as f:
        loaded_data = pkl.load(f)
        bl_dataset = loaded_data
    with open(br_path, 'rb') as f:
        loaded_data = pkl.load(f)
        br_dataset = loaded_data

    with open(pf_path, 'rb') as f:
        loaded_data = pkl.load(f)
        pf_dataset = loaded_data
    with open(pb_path, 'rb') as f:
        loaded_data = pkl.load(f)
        pb_dataset = loaded_data
    with open(pl_path, 'rb') as f:
        loaded_data = pkl.load(f)
        pl_dataset = loaded_data
    with open(pr_path, 'rb') as f:
        loaded_data = pkl.load(f)
        pr_dataset = loaded_data

    dataset = {}
    for key in tf_dataset.keys():
        dataset[key] = np.concatenate((tf_dataset[key], tb_dataset[key], tl_dataset[key], tr_dataset[key],
                                            bf_dataset[key], bf_dataset[key], bf_dataset[key], bf_dataset[key],
                                            pace_dataset[key])
                                      , axis=0)
    ###############################################################
    dataset = preprocess_fn(dataset)

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = 'timeouts' in dataset

    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)

        for k in dataset:
            if 'metadata' in k: continue
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            if 'maze2d' in env.name:
                episode_data = process_maze2d_episode(episode_data)
            yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1


#-----------------------------------------------------------------------------#
#-------------------------------- maze2d fixes -------------------------------#
#-----------------------------------------------------------------------------#

def process_maze2d_episode(episode):
    '''
        adds in `next_observations` field to episode
    '''
    assert 'next_observations' not in episode
    length = len(episode['observations'])
    next_observations = episode['observations'][1:].copy()
    for key, val in episode.items():
        episode[key] = val[:-1]
    episode['next_observations'] = next_observations
    return episode
