from prompt_dt.prompt_utils import get_env_list
import os
from collections import namedtuple
import json, pickle, os

cur_dir = os.getcwd()
config_save_path = os.path.join(cur_dir, 'config')
data_save_path = os.path.join(cur_dir, 'data')
save_path = os.path.join(cur_dir, 'model_saved/')
if not os.path.exists(save_path): os.mkdir(save_path)

config_path_dict = {
    'cheetah_vel': "cheetah_vel/cheetah_vel_40.json",
    'cheetah_dir': "cheetah_dir/cheetah_dir_2.json",
    'ant_dir': "ant_dir/ant_dir_50.json",
    'ML1-pick-place-v2': "ML1-pick-place-v2/ML1_pick_place.json",
}

task_config = os.path.join(config_save_path, config_path_dict['cheetah_dir'])
with open(task_config, 'r') as f:
    task_config = json.load(f, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
train_env_name_list, test_env_name_list = [], []
for task_ind in task_config.train_tasks:
    train_env_name_list.append('cheetah_dir' +'-'+ str(task_ind))
for task_ind in task_config.test_tasks:
    test_env_name_list.append('cheetah_dir' +'-'+ str(task_ind))
# training envs
info, env_list = get_env_list(train_env_name_list, config_save_path, 'cuda:0')

