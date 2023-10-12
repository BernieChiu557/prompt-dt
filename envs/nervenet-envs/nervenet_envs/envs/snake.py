import sys

from collections import namedtuple
import json, pickle, os
import os.path as osp
import numpy as np
from typing import List

# import init_path
import num2words

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


_this_dir = osp.dirname(__file__)
_base_dir = osp.join(_this_dir, '..')
add_path(_base_dir)

# DEFAULT_CAMERA_CONFIG = {
#     "trackbodyid": 0,
#     "distance": 10.0,
#     "elevation": -20.0,
# }
DEFAULT_CAMERA_CONFIG = {
    "distance": 6.0,
}

class SnakeEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(self, pod_number=3, is_crippled=False, reset_noise_scale=0.1, **kwargs):
        utils.EzPickle.__init__(
            self,
            pod_number,
            is_crippled,
            reset_noise_scale,
            **kwargs
        )

        # get the path of the environments
        if is_crippled:
            xml_name = 'CrippledSnake' + self.get_env_num_str(pod_number) + \
                '.xml'
        else:
            xml_name = 'Snake' + self.get_env_num_str(pod_number) + '.xml'
        xml_path = os.path.join(os.path.join(_base_dir, 'envs', 'assets', xml_name))
        xml_path = str(os.path.abspath(xml_path))

        self.num_body = pod_number
        self._direction = 0
        self.ctrl_cost_coeff = 0.0001 / pod_number * 3
        self._reset_noise_scale = reset_noise_scale
        self.step_count = 0

        # fake observation space
        obs_shape = 5
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
        )

        MujocoEnv.__init__(
            self,
            xml_path,
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        obs_shape = len(self._get_obs())
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
        )
    
    def get_env_num_str(self, number):
        num_str = num2words.num2words(number)
        return num_str[0].upper() + num_str[1:]

    def step(self, action):
        xposbefore = self.get_body_com("podBody_1")[self._direction].copy()
        self.do_simulation(action, self.frame_skip)
        xposafter = self.get_body_com("podBody_1")[self._direction].copy()
        
        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = - self.ctrl_cost_coeff * np.square(action).sum()
        reward = reward_fwd + reward_ctrl
        
        observation = self._get_obs()

        self.step_count += 1
        if self.step_count == 1000:
            terminated = True
        else:
            terminated = False
        info = {
            "reward_forward": reward_fwd,
            "reward_ctrl": reward_ctrl,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        return np.concatenate((position[2:], velocity))

    def reset_model(self):
        # breakpoint()
        self.step_count = 0
        self.set_state(
            self.init_qpos + self.np_random.uniform(
                low=-.1, high=.1, size=self.model.nq
            ),
            self.init_qvel + self.np_random.uniform(
                low=-.1, high=.1, size=self.model.nv
            )
        )
        # breakpoint()
        return self._get_obs()

class MultitaskSnakeEnv(SnakeEnv):
    def __init__(self, task={}, n_tasks=2, **kwargs):
        self._task = task
        self.tasks = self.sample_tasks(n_tasks)
        self._goal = self.tasks[0]['goal']
        super().__init__(**kwargs)

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task['goal'] # assume parameterization of task by single vector
        self.reset()

class SnakeDirEnv_(MultitaskSnakeEnv):

    def __init__(self, task={}, n_tasks=2, forward_backward=False, randomize_tasks=True, **kwargs):
        self.forward_backward = forward_backward
        super(SnakeDirEnv_, self).__init__(task, n_tasks, **kwargs)

    def step(self, action):
        torso_xyz_before = np.array(self.get_body_com("podBody_1"))

        direct = (np.cos(self._goal), np.sin(self._goal))

        self.do_simulation(action, self.frame_skip)

        torso_xyz_after = np.array(self.get_body_com("podBody_1"))
        torso_velocity = torso_xyz_after - torso_xyz_before

        forward_reward = np.dot((torso_velocity[:2]/self.dt), direct)
        ctrl_cost = self.ctrl_cost_coeff * np.square(action).sum()

        reward = forward_reward - ctrl_cost

        self.step_count += 1
        if self.step_count == 1000:
            terminated = True
        else:
            terminated = False
        ob = self._get_obs()

        if self.render_mode == "human":
            self.render()

        return ob, reward, terminated, False, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
        )

    def sample_tasks(self, num_tasks):
        if self.forward_backward:
            assert num_tasks == 2
            velocities = np.array([0., np.pi])
        else:
            velocities = np.random.uniform(0., 2.0 * np.pi, size=(num_tasks,))
        tasks = [{'goal': velocity} for velocity in velocities]
        return tasks

class SnakeDirEnv(SnakeDirEnv_):
    def __init__(self, tasks: List[dict], n_tasks: int = None, include_goal: bool = False, **kwargs):
        self.include_goal = include_goal
        super(SnakeDirEnv, self).__init__(forward_backward=n_tasks == 2, **kwargs)
        if tasks is None:
            assert n_tasks is not None, "Either tasks or n_tasks must be non-None"
            tasks = self.sample_tasks(n_tasks)
        self.tasks = tasks
        self.n_tasks = len(self.tasks)
        self.set_task_idx(0)
        print(f'goal direction in rad: {self._goal}')
        self._max_episode_steps = 1000
    
    def _get_obs(self):
        if self.include_goal:
            idx = 0
            try:
                idx = self.tasks.index(self._task)
            except:
                pass
            one_hot = np.zeros(50, dtype=np.float32)
            one_hot[idx] = 1.0
            obs = super()._get_obs()
            obs = np.concatenate([obs, one_hot])
        else:
            obs = super()._get_obs()
        return obs
    
    def set_task(self, task):
        self._task = task
        self._goal = task['goal']
        self.reset()

    def set_task_idx(self, idx):
        self.set_task(self.tasks[idx])
