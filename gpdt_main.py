import gymnasium
import numpy as np
import torch
import wandb

import argparse
import pickle
import random
import sys
import time
import itertools

from gpdt.prompt_dt_mmml import Prompt_DT_MMML
from gpdt.gpdt import GPDT_V2, GPDT_V3

from gpdt.gpdt_seq_trainer import GPDTSequenceTrainer
from gpdt.gpdt_utils import get_env_list
from gpdt.gpdt_utils import get_prompt_batch, get_prompt, get_batch, get_batch_finetune
from gpdt.gpdt_utils import load_data_prompt, process_info # process_total_data_mean, 
from gpdt.gpdt_utils import eval_episodes
from gpdt.gpdt_utils import get_agent_config

from collections import namedtuple
import json, pickle, os

# python gpdt_main.py --env snake_dir_MMMT --model_type gpdt_v2 --device cuda:0
# python gpdt_main.py --env snake_dir_3_to_4 --model_type gpdt_v2
# python gpdt_main.py --env snake_dir_3_to_4 --model_type pdt --finetune --device cuda:1 --finetune_batch_size 64


def experiment_mix_env(
        exp_prefix,
        variant,
):
    device = variant['device']
    log_to_wandb = variant['log_to_wandb']
    if variant['model_type'] == 'gpdt_v2' or variant['model_type'] == 'gpdt_v3':
        variant['shift_action'] = True
    else:
        variant['shift_action'] = False

    ######
    # construct train and test environments
    ######
    
    cur_dir = os.getcwd()
    config_save_path = os.path.join(cur_dir, 'config')
    data_save_path = os.path.join(cur_dir, 'data')
    save_path = os.path.join(cur_dir, 'model_saved/')
    if not os.path.exists(save_path): os.mkdir(save_path)

    config_path_dict = {
        'snake_dir_3_to_4': "snake_dir/snake_3_to_4_10.json",
        'snake_dir_MMMT': "snake_dir/snake_MMMT_45.json",
        'centipede_4_to_6': "centipede_dir/centipede_4_to_6_10.json",
        'centipede_MMMT': "centipede_dir/centipede_MMMT_45.json"
    }
    if args.env == 'snake_dir_3_to_4':
        agents = {}
        for length in [3, 4]:
            agent = get_agent_config(f'snake{length}')
            agents[length] = agent
    elif args.env == 'snake_dir_MMMT':
        agents = {}
        for length in [3, 4, 5, 6, 7]:
            agent = get_agent_config(f'snake{length}')
            agents[length] = agent
    else:
        print(f'{args.env} not implement yet in agent_config, ignore if using original pdt')
        # raise NotImplementedError
    
    task_config = os.path.join(config_save_path, config_path_dict[args.env])
    with open(task_config, 'r') as f:
        task_config = json.load(f, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
    train_env_name_list, test_env_name_list = [], []
    # length means pod_number for snake and leg_number for centipede
    for length, task_ind in zip(task_config.train_tasks.lengths, task_config.train_tasks.task_inds):
        train_env_name_list.append(args.env +'-' + str(length) +'-'+ str(task_ind))
    for length, task_ind in zip(task_config.test_tasks.lengths, task_config.test_tasks.task_inds):
        test_env_name_list.append(args.env +'-' + str(length) +'-'+ str(task_ind))
    # training envs
    info, env_list = get_env_list(train_env_name_list, config_save_path, device)
    # testing envs
    test_info, test_env_list = get_env_list(test_env_name_list, config_save_path, device)

    print(f'Env Info: {info} \n\n Test Env Info: {test_info}\n\n\n')
    print(f'Env List: {env_list} \n\n Test Env List: {test_env_list}')

    ######
    # process train and test datasets
    ######

    K = variant['K']
    batch_size = variant['batch_size']
    pct_traj = variant.get('pct_traj', 1.)
    mode = variant.get('mode', 'normal')
    dataset_mode = variant['dataset_mode']
    test_dataset_mode = variant['test_dataset_mode']
    train_prompt_mode = variant['train_prompt_mode']
    test_prompt_mode = variant['test_prompt_mode']

    # load training dataset
    trajectories_list, prompt_trajectories_list = load_data_prompt(train_env_name_list, data_save_path, dataset_mode, train_prompt_mode)
    # load testing dataset
    test_trajectories_list, test_prompt_trajectories_list = load_data_prompt(test_env_name_list, data_save_path, test_dataset_mode, test_prompt_mode)

    # process train info
    info = process_info(train_env_name_list, trajectories_list, info, mode, dataset_mode, pct_traj, variant)
    # process test info
    test_info = process_info(test_env_name_list, test_trajectories_list, test_info, mode, test_dataset_mode, pct_traj, variant)

    ######
    # construct gpdt model and trainer
    ######
    exp_prefix = exp_prefix + '-' + args.env
    num_env = len(train_env_name_list)
    group_name = f'{exp_prefix}-{str(num_env)}-Env-{dataset_mode}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    if variant['model_type'] == 'pdt':
        model = Prompt_DT_MMML(
            agents=agents,
            max_length=K,
            max_ep_len=1000,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4 * variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
    elif variant['model_type'] == 'gpdt_v2':
        model = GPDT_V2(
            agents=agents,
            max_length=K,
            max_ep_length=1000,
            prompt_length=variant['prompt_length'],
            hidden_size=variant['embed_dim'],
            n_layers=variant['n_layer'],
            dropout=variant['dropout'],
        )
    elif variant['model_type'] == 'gpdt_v3':
        model = GPDT_V3(
            agents=agents,
            max_length=K,
            max_ep_length=1000,
            prompt_length=variant['prompt_length'],
            hidden_size=variant['embed_dim'],
            n_layers=variant['n_layer'],
            dropout=variant['dropout'],
        )
    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / warmup_steps, 1)
    )


    env_name = train_env_name_list[0]
    trainer = GPDTSequenceTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        get_batch=get_batch(trajectories_list[0], info[env_name], variant),
        scheduler=scheduler,
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
        eval_fns=None,
        get_prompt=get_prompt(prompt_trajectories_list[0], info[env_name], variant),
        get_prompt_batch=get_prompt_batch(trajectories_list, prompt_trajectories_list, info, variant, train_env_name_list),
        get_prompt_batch_finetune=get_prompt_batch(test_trajectories_list, test_prompt_trajectories_list, test_info, variant, test_env_name_list),
        shift_action=variant['shift_action'],
    )


    if not variant['evaluation']:
        ######
        # start training
        ######
        if log_to_wandb:
            wandb.init(
                name=exp_prefix,
                group=group_name,
                project='graph-prompt-decision-transformer',
                config=variant
            )
            save_path += wandb.run.name
            os.mkdir(save_path)

        # construct model post fix
        model_post_fix = '_TRAIN_'+variant['train_prompt_mode']+'_TEST_'+variant['test_prompt_mode']
        # if variant['no_prompt']:
        #     model_post_fix += '_NO_PROMPT'
        if variant['finetune']:
            model_post_fix += '_FINETUNE'
        # if variant['no_r']:
        #     model_post_fix += '_NO_R'
        
        for iter in range(variant['max_iters']):
            # env_id = iter % num_env
            # env_name = train_env_name_list[env_id]
            # breakpoint()
            outputs = trainer.pure_train_iteration_mix(
                num_steps=variant['num_steps_per_iter'], 
                no_prompt=args.no_prompt
                )

            # start evaluation
            if iter % args.test_eval_interval == 0:
                # evaluate test
                if not args.finetune:
                    test_eval_logs = trainer.eval_iteration_multienv(
                        get_prompt, test_prompt_trajectories_list,
                        eval_episodes, test_env_name_list, test_info, variant, test_env_list, iter_num=iter + 1, 
                        print_logs=True, no_prompt=args.no_prompt, group='test')
                    outputs.update(test_eval_logs)
                else:
                    pass
                    test_eval_logs = trainer.finetune_eval_iteration_multienv(
                        get_prompt, get_batch_finetune, test_prompt_trajectories_list, test_trajectories_list,
                        eval_episodes, test_env_name_list, test_info, 
                        variant, test_env_list, iter_num=iter + 1, 
                        print_logs=True, no_prompt=args.no_prompt, 
                        group='finetune-test', finetune_opt=variant['finetune_opt'])
                    outputs.update(test_eval_logs)
            
            if iter % args.train_eval_interval == 0:
                # evaluate train
                train_eval_logs = trainer.eval_iteration_multienv(
                    get_prompt, prompt_trajectories_list,
                    eval_episodes, train_env_name_list, info, variant, env_list, iter_num=iter + 1, 
                    print_logs=True, no_prompt=args.no_prompt, group='train')
                outputs.update(train_eval_logs)

            if iter % variant['save_interval'] == 0:
                trainer.save_model(
                    env_name=args.env, 
                    postfix=model_post_fix+'_iter_'+str(iter), 
                    folder=save_path)

            outputs.update({"global_step": iter}) # set global step as iteration

            if log_to_wandb:
                wandb.log(outputs)
        
        trainer.save_model(env_name=args.env,  postfix=model_post_fix+'_iter_'+str(iter),  folder=save_path)

    else:
        ####
        # start evaluating
        ####
        saved_model_path = os.path.join(save_path, variant['load_path'])
        model.load_state_dict(torch.load(saved_model_path))
        print('model initialized from: ', saved_model_path)
        eval_iter_num = int(saved_model_path.split('_')[-1])

        eval_logs = trainer.eval_iteration_multienv(
                    get_prompt, test_prompt_trajectories_list,
                    eval_episodes, test_env_name_list, test_info, variant, test_env_list, iter_num=eval_iter_num, 
                    print_logs=True, no_prompt=args.no_prompt, group='eval')

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='snake_dir_MMMT') # ['cheetah_dir', 'cheetah_vel', 'ant_dir', 'ML1-pick-place-v2']
    parser.add_argument('--dataset_mode', type=str, default='expert')
    parser.add_argument('--test_dataset_mode', type=str, default='expert')
    parser.add_argument('--train_prompt_mode', type=str, default='expert')
    parser.add_argument('--test_prompt_mode', type=str, default='expert')

    parser.add_argument('--prompt-episode', type=int, default=1)
    parser.add_argument('--prompt-length', type=int, default=5)
    parser.add_argument('--stochastic-prompt', action='store_true', default=True)
    parser.add_argument('--no-prompt', action='store_true', default=False)
    parser.add_argument('--no-r', action='store_true', default=False)
    parser.add_argument('--no-rtg', action='store_true', default=False)
    parser.add_argument('--finetune', action='store_true', default=False)
    parser.add_argument('--finetune_steps', type=int, default=10)
    parser.add_argument('--finetune_batch_size', type=int, default=256)
    parser.add_argument('--finetune_opt', action='store_true', default=True)
    parser.add_argument('--finetune_lr', type=float, default=1e-4)
    parser.add_argument('--no_state_normalize', action='store_true', default=False) 
    parser.add_argument('--evaluation', action='store_true', default=False) 
    parser.add_argument('--render', action='store_true', default=False) 
    parser.add_argument('--load-path', type=str, default= None) # choose a model when in evaluation mode

    parser.add_argument('--mode', type=str, default='normal')
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=5)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000) # 10000*(number of environments)
    parser.add_argument('--num_eval_episodes', type=int, default=50) 
    parser.add_argument('--max_iters', type=int, default=5000) 
    parser.add_argument('--num_steps_per_iter', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=True)
    parser.add_argument('--train_eval_interval', type=int, default=500)
    parser.add_argument('--test_eval_interval', type=int, default=100)
    parser.add_argument('--save-interval', type=int, default=500)
    
    parser.add_argument('--model_type', type=str, default='gpdt_v2')

    args = parser.parse_args()
    experiment_mix_env('nervenet-experiment', variant=vars(args))