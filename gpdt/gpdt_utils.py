import numpy as np
# import gym
import gymnasium as gym
import json, pickle, random, os, torch
from collections import namedtuple
from .gpdt_evaluate_episodes import prompt_evaluate_episode, prompt_evaluate_episode_rtg

import nervenet_envs


""" constructing envs """

def gen_env(env_name, config_save_path):
    if 'snake_dir' in env_name:
        length = int(env_name.split('-')[-2])
        task_idx = int(env_name.split('-')[-1])
        task_paths = f"{config_save_path}/snake_dir/config_snake{length}_dir_task{task_idx}.pkl"
        tasks = []
        with open(task_paths.format(task_idx), 'rb') as f:
            task_info = pickle.load(f)
            assert len(task_info) == 1, f'Unexpected task info: {task_info}'
            tasks.append(task_info[0])
        env = gym.make('SnakeDir-v0', pod_number=length, tasks=tasks, n_tasks=len(tasks), include_goal = False)
        max_ep_len = 1000

        # TODO test different target
        env_targets = [500]
        scale = 500.
    else:
        raise NotImplementedError
    return env, max_ep_len, env_targets, scale


def get_env_list(env_name_list, config_save_path, device):
    info = {} # store all the attributes for each env
    env_list = []
    
    for env_name in env_name_list:
        info[env_name] = {}
        env, max_ep_len, env_targets, scale = gen_env(env_name=env_name, config_save_path=config_save_path)
        info[env_name]['max_ep_len'] = max_ep_len
        info[env_name]['env_targets'] = env_targets
        info[env_name]['scale'] = scale
        info[env_name]['state_dim'] = env.observation_space.shape[0]
        info[env_name]['act_dim'] = env.action_space.shape[0] 
        info[env_name]['device'] = device
        env_list.append(env)
    return info, env_list


""" prompts """

def flatten_prompt(prompt, batch_size):
    p_s, p_a, p_r, p_d, p_rtg, p_timesteps, p_mask = prompt
    p_s = p_s.reshape((batch_size, -1, p_s.shape[-1]))
    p_a = p_a.reshape((batch_size, -1, p_a.shape[-1]))
    p_r = p_r.reshape((batch_size, -1, p_r.shape[-1]))
    p_d = p_d.reshape((batch_size, -1))
    p_rtg = p_rtg[:,:-1,:]
    p_rtg = p_rtg.reshape((batch_size, -1, p_rtg.shape[-1]))
    p_timesteps = p_timesteps.reshape((batch_size, -1))
    p_mask = p_mask.reshape((batch_size, -1)) 
    return p_s, p_a, p_r, p_d, p_rtg, p_timesteps, p_mask


def get_prompt(prompt_trajectories, info, variant):
    num_trajectories, p_sample, sorted_inds = info['num_trajectories'], info['p_sample'], info['sorted_inds']
    max_ep_len, state_mean, state_std, scale = info['max_ep_len'], info['state_mean'], info['state_std'], info['scale']
    state_dim, act_dim, device = info['state_dim'], info['act_dim'], info['device']
    max_state_dim, max_act_dim = info['max_state_dim'], info['max_act_dim']
    num_episodes, max_len = variant['prompt_episode'], variant['prompt_length']
    shift_action = variant['shift_action']
    # breakpoint()

    def fn(sample_size=1):
        # random sample prompts with fixed length (prompt-length) in num episodes (prompt-episode)
        batch_inds = np.random.choice(
            np.arange(len(prompt_trajectories)),
            size=int(num_episodes*sample_size),
            replace=True,
            # p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(int(num_episodes*sample_size)):
            if variant["stochastic_prompt"]:
                traj = prompt_trajectories[int(batch_inds[i])] # random select traj
            else:
                traj = prompt_trajectories[int(sorted_inds[-i])] # select the best traj with highest rewards
                # traj = prompt_trajectories[i]
            si = max(0, traj['rewards'].shape[0] - max_len -1) # select the last traj with length max_len

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            if shift_action and si > 0:
                a.append(traj['actions'][si-1:si + max_len].reshape(1, -1, act_dim))
            else:
                a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            alen = a[-1].shape[1]
            # if tlen !=args.K:
            #     print('tlen not equal to k')
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            if not variant['no_state_normalize']:
                s[-1] = (s[-1] - state_mean) / state_std
            if shift_action:
                a[-1] = np.concatenate([np.ones((1, 1 + max_len - alen, act_dim)) * -10., a[-1]], axis=1)
            else:
                a[-1] = np.concatenate([np.ones((1, max_len - alen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0))
        s_pad = torch.zeros(s.shape[0], s.shape[1], max_state_dim - state_dim)
        s = torch.cat([s, s_pad], dim=2).to(dtype=torch.float32, device=device)

        a = torch.from_numpy(np.concatenate(a, axis=0))
        a_pad = torch.zeros(a.shape[0], a.shape[1], max_act_dim - act_dim)
        a = torch.cat([a, a_pad], dim=2).to(dtype=torch.float32, device=device)

        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
        return s, a, r, d, rtg, timesteps, mask

    return fn


def get_prompt_batch(trajectories_list, prompt_trajectories_list, info, variant, train_env_name_list):
    per_env_batch_size = variant['batch_size']
 
    def fn(batch_size=per_env_batch_size):
        p_s_list, p_a_list, p_r_list, p_d_list, p_rtg_list, p_timesteps_list, p_mask_list = [], [], [], [], [], [], []
        s_list, a_list, r_list, d_list, rtg_list, timesteps_list, mask_list = [], [], [], [], [], [], []
        per_env_b_size = batch_size
        lengths = []
        act_dims = []
        for env_id, env_name in enumerate(train_env_name_list):
            length = int(env_name.split('-')[-2])
            lengths.append(length)
            act_dim = info[env_name]['act_dim']
            act_dims.append(act_dim)

            if prompt_trajectories_list:
                get_prompt_fn = get_prompt(prompt_trajectories_list[env_id], info[env_name], variant)
            else:
                get_prompt_fn = get_prompt(trajectories_list[env_id], info[env_name], variant)
            get_batch_fn = get_batch(trajectories_list[env_id], info[env_name], variant) 
            prompt = flatten_prompt(get_prompt_fn(batch_size), batch_size)
            p_s, p_a, p_r, p_d, p_rtg, p_timesteps, p_mask = prompt
            p_s_list.append(p_s)
            p_a_list.append(p_a)
            p_r_list.append(p_r)
            p_d_list.append(p_d)
            p_rtg_list.append(p_rtg)
            p_timesteps_list.append(p_timesteps)
            p_mask_list.append(p_mask)

            batch = get_batch_fn(batch_size=batch_size)
            s, a, r, d, rtg, timesteps, mask = batch
            if variant['no_r']:
                r = torch.zeros_like(r)
            if variant['no_rtg']:
                rtg = torch.zeros_like(rtg)
            s_list.append(s)
            a_list.append(a)
            r_list.append(r)
            d_list.append(d)
            rtg_list.append(rtg)
            timesteps_list.append(timesteps)
            mask_list.append(mask)

        p_s, p_a, p_r, p_d = torch.cat(p_s_list, dim=0), torch.cat(p_a_list, dim=0), torch.cat(p_r_list, dim=0), torch.cat(p_d_list, dim=0)
        p_rtg, p_timesteps, p_mask = torch.cat(p_rtg_list, dim=0), torch.cat(p_timesteps_list, dim=0), torch.cat(p_mask_list, dim=0)
        s, a, r, d = torch.cat(s_list, dim=0), torch.cat(a_list, dim=0), torch.cat(r_list, dim=0), torch.cat(d_list, dim=0)
        rtg, timesteps, mask = torch.cat(rtg_list, dim=0), torch.cat(timesteps_list, dim=0), torch.cat(mask_list, dim=0)
        prompt = p_s, p_a, p_r, p_d, p_rtg, p_timesteps, p_mask
        batch = s, a, r, d, rtg, timesteps, mask

        # breakpoint()
        num = lengths.count(lengths[0])
        lengths = list(set(lengths))
        act_dims = list(set(act_dims))
        per_env_b_size = per_env_b_size * num

        return per_env_b_size, lengths, act_dims, prompt, batch
    return fn

""" batches """

def get_batch(trajectories, info, variant):
    num_trajectories, p_sample, sorted_inds = info['num_trajectories'], info['p_sample'], info['sorted_inds']
    max_ep_len, state_mean, state_std, scale = info['max_ep_len'], info['state_mean'], info['state_std'], info['scale']
    state_dim, act_dim, device = info['state_dim'], info['act_dim'], info['device']
    # breakpoint()
    max_state_dim, max_act_dim = info['max_state_dim'], info['max_act_dim']
    batch_size, K = variant['batch_size'], variant['K']
    shift_action = variant['shift_action']

    def fn(batch_size=batch_size, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            if shift_action and si > 0:
                a.append(traj['actions'][si-1:si + max_len].reshape(1, -1, act_dim))
            else:
                a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            alen = a[-1].shape[1]
            # if tlen !=args.K:
            #     print('tlen not equal to k')
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            if not variant['no_state_normalize']:
                s[-1] = (s[-1] - state_mean) / state_std
            if shift_action:
                a[-1] = np.concatenate([np.ones((1, 1 + max_len - alen, act_dim)) * -10., a[-1]], axis=1)
            else:
                a[-1] = np.concatenate([np.ones((1, max_len - alen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0))
        s_pad = torch.zeros(s.shape[0], s.shape[1], max_state_dim - state_dim)
        s = torch.cat([s, s_pad], dim=2).to(dtype=torch.float32, device=device)

        a = torch.from_numpy(np.concatenate(a, axis=0))
        a_pad = torch.zeros(a.shape[0], a.shape[1], max_act_dim - act_dim)
        a = torch.cat([a, a_pad], dim=2).to(dtype=torch.float32, device=device)

        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device) # TODO: why mask only has several zeros

        return s, a, r, d, rtg, timesteps, mask

    return fn


def get_batch_finetune(trajectories, info, variant):
    num_trajectories, p_sample, sorted_inds = info['num_trajectories'], info['p_sample'], info['sorted_inds']
    max_ep_len, state_mean, state_std, scale = info['max_ep_len'], info['state_mean'], info['state_std'], info['scale']
    state_dim, act_dim, device = info['state_dim'], info['act_dim'], info['device']
    max_state_dim, max_act_dim = info['max_state_dim'], info['max_act_dim']
    batch_size, K = variant['batch_size'], variant['prompt_length'] # use the same amount of data for funetuning
    shift_action = variant['shift_action']

    def fn(batch_size=batch_size, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)
            si = max(0, traj['rewards'].shape[0] - max_len -1) # select the last traj with length max_len

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            if shift_action and si > 0:
                a.append(traj['actions'][si-1:si + max_len].reshape(1, -1, act_dim))
            else:
                a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            alen = a[-1].shape[1]
            # if tlen !=args.K:
            #     print('tlen not equal to k')
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            if not variant['no_state_normalize']:
                s[-1] = (s[-1] - state_mean) / state_std
            if shift_action:
                a[-1] = np.concatenate([np.ones((1, 1 + max_len - alen, act_dim)) * -10., a[-1]], axis=1)
            else:
                a[-1] = np.concatenate([np.ones((1, max_len - alen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0))
        s_pad = torch.zeros(s.shape[0], s.shape[1], max_state_dim - state_dim)
        s = torch.cat([s, s_pad], dim=2)
        s.to(dtype=torch.float32, device=device)

        a = torch.from_numpy(np.concatenate(a, axis=0))
        a_pad = torch.zeros(s.shape[0], s.shape[1], max_act_dim - act_dim)
        a = torch.cat([a, a_pad], dim=2)
        a.to(dtype=torch.float32, device=device)
        
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device) # TODO: why mask only has several zeros

        return s, a, r, d, rtg, timesteps, mask

    return fn

""" data processing """

def process_dataset(trajectories, mode, env_name, dataset, pct_traj):
    # save all path information into separate lists
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)
    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj * num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])
    reward_info = [np.mean(returns), np.std(returns), np.max(returns), np.min(returns)]

    return trajectories, num_trajectories, sorted_inds, p_sample, state_mean, state_std, reward_info


def load_data_prompt(env_name_list, data_save_path, dataset, prompt_mode):
    trajectories_list = []
    prompt_trajectories_list = []
    for env_name in env_name_list:
        agent = env_name.split('_')[0]
        task = env_name.split('_')[1]
        length = int(env_name.split('-')[-2])
        task_idx = int(env_name.split('-')[-1])

        dataset_path = data_save_path+f'/{agent}_{task}/{agent}{length}_{task}-{task_idx}-{dataset}.pkl'
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
        prompt_dataset_path = data_save_path+f'/{agent}_{task}/{agent}{length}_{task}-{task_idx}-prompt-{prompt_mode}.pkl'
        with open(prompt_dataset_path, 'rb') as f:
            prompt_trajectories = pickle.load(f)
        trajectories_list.append(trajectories)
        prompt_trajectories_list.append(prompt_trajectories)
    
    return trajectories_list, prompt_trajectories_list


def process_info(env_name_list, trajectories_list, info, mode, dataset, pct_traj, variant):
    max_state_dim = 0
    max_act_dim = 0
    for i, env_name in enumerate(env_name_list):
        trajectories, num_trajectories, sorted_inds, p_sample, state_mean, state_std, reward_info = process_dataset(
            trajectories=trajectories_list[i], mode=mode, env_name=env_name_list[i], dataset=dataset, pct_traj=pct_traj)
        info[env_name]['num_trajectories'] = num_trajectories
        info[env_name]['sorted_inds'] = sorted_inds
        info[env_name]['p_sample'] = p_sample
        info[env_name]['state_mean'] = state_mean
        info[env_name]['state_std'] = state_std

        state_dim = info[env_name]['state_dim'] 
        act_dim = info[env_name]['act_dim']
        if state_dim > max_state_dim and act_dim > max_act_dim:
            max_state_dim = state_dim
            max_act_dim = act_dim
            for env_name_ in env_name_list:
                info[env_name_]['max_state_dim'] = max_state_dim
                info[env_name_]['max_act_dim'] = max_act_dim

    print('-'*50)
    print(f'max_state_dim: {max_state_dim}')
    print(f'max_act_dim: {max_act_dim}')
    print('-'*50)
    
    return info


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum

""" evaluation """

def eval_episodes(target_rew, info, variant, env, env_name):
    max_ep_len, state_mean, state_std, scale = info['max_ep_len'], info['state_mean'], info['state_std'], info['scale']
    state_dim, act_dim, device = info['state_dim'], info['act_dim'], info['device']
    num_eval_episodes = variant['num_eval_episodes']
    mode = variant.get('mode', 'normal')
    length = int(env_name.split('-')[-2])

    def fn(model, prompt=None):
        returns = []
        for _ in range(num_eval_episodes):
            with torch.no_grad():
                ret, infos = prompt_evaluate_episode_rtg(
                    env,
                    length,
                    state_dim,
                    act_dim,
                    model,
                    max_ep_len=max_ep_len,
                    scale=scale,
                    target_return=target_rew / scale,
                    mode=mode,
                    state_mean=state_mean,
                    state_std=state_std,
                    device=device,
                    prompt=prompt,
                    no_r=variant['no_r'],
                    no_rtg=variant['no_rtg'],
                    no_state_normalize=variant['no_state_normalize']                
                    )
            returns.append(ret)
        return {
            f'{env_name}_target_{target_rew}_return_mean': np.mean(returns),
            f'{env_name}_target_{target_rew}_return_std': np.std(returns),
            }
    return fn

def get_agent_config(name):
    if name == 'snake3':
        agent = {
            'state_dim': 8,
            'act_dim': 2,
            'num_node': 3,
            'node_type_state_action_len': {
                'root': 4,
                'seg': 3,
            },
            'node_type': {
                'root': 'root',
                'seg_1': 'seg',
                'seg_2': 'seg',
            },
            'state_position': {
                'root': [0, 3, 4, 5],
                'seg_1': [1, 6],
                'seg_2': [2, 7],
            },
            'action_position': {
                'root': [],
                'seg_1': [0],
                'seg_2': [1],
            },
            'edge': [[0, 1], [1, 2]],
        }
    elif name == 'snake4':
        agent = {
            'state_dim': 10,
            'act_dim': 3,
            'num_node': 4,
            'node_type_state_action_len': {
                'root': 4,
                'seg': 3,
            },
            'node_type': {
                'root': 'root',
                'seg_1': 'seg',
                'seg_2': 'seg',
                'seg_3': 'seg',
            },
            'state_position': {
                'root': [0, 4, 5, 6],
                'seg_1': [1, 7],
                'seg_2': [2, 8],
                'seg_3': [3, 9],
            },
            'action_position': {
                'root': [],
                'seg_1': [0],
                'seg_2': [1],
                'seg_3': [2],
            },
            'edge': [[0, 1], [1, 2], [2, 3]],
        }
    elif name == 'snake5':
        agent = {
            'state_dim': 12,
            'act_dim': 4,
            'num_node': 5,
            'node_type_state_action_len': {
                'root': 4,
                'seg': 3,
            },
            'node_type': {
                'root': 'root',
                'seg_1': 'seg',
                'seg_2': 'seg',
                'seg_3': 'seg',
                'seg_4': 'seg',
            },
            'state_position': {
                'root': [0, 5, 6, 7],
                'seg_1': [1, 8],
                'seg_2': [2, 9],
                'seg_3': [3, 10],
                'seg_4': [4, 11],
            },
            'action_position': {
                'root': [],
                'seg_1': [0],
                'seg_2': [1],
                'seg_3': [2],
                'seg_4': [3],
            },
            'edge': [[0, 1], [1, 2], [2, 3], [3, 4]],
        }
    elif name == 'snake6':
        agent = {
            'state_dim': 14,
            'act_dim': 5,
            'num_node': 6,
            'node_type_state_action_len': {
                'root': 4,
                'seg': 3,
            },
            'node_type': {
                'root': 'root',
                'seg_1': 'seg',
                'seg_2': 'seg',
                'seg_3': 'seg',
                'seg_4': 'seg',
                'seg_5': 'seg',
            },
            'state_position': {
                'root': [0, 6, 7, 8],
                'seg_1': [1, 9],
                'seg_2': [2, 10],
                'seg_3': [3, 11],
                'seg_4': [4, 12],
                'seg_5': [5, 13],
            },
            'action_position': {
                'root': [],
                'seg_1': [0],
                'seg_2': [1],
                'seg_3': [2],
                'seg_4': [3],
                'seg_5': [4],
            },
            'edge': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]],
        }
    elif name == 'snake7':
        agent = {
            'state_dim': 16,
            'act_dim': 6,
            'num_node': 7,
            'node_type_state_action_len': {
                'root': 4,
                'seg': 3,
            },
            'node_type': {
                'root': 'root',
                'seg_1': 'seg',
                'seg_2': 'seg',
                'seg_3': 'seg',
                'seg_4': 'seg',
                'seg_5': 'seg',
                'seg_6': 'seg',
            },
            'state_position': {
                'root': [0, 7, 8, 9],
                'seg_1': [1, 10],
                'seg_2': [2, 11],
                'seg_3': [3, 12],
                'seg_4': [4, 13],
                'seg_5': [5, 14],
                'seg_6': [6, 15],
            },
            'action_position': {
                'root': [],
                'seg_1': [0],
                'seg_2': [1],
                'seg_3': [2],
                'seg_4': [3],
                'seg_5': [4],
                'seg_6': [5],
            },
            'edge': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]],
        }
    else:
        raise NameError('agent name not valid, only "hopper", "halfcheetah" and "walker2d" are implemented')
    
    return agent