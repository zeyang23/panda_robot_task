import time
import numpy as np

import gym
import panda_gym
import custom_envs

from stable_baselines3 import PPO

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

from typing import Callable


# def make_env(env_id: str, rank: int, seed: int = 0) -> Callable:
#     """
#     Utility function for multiprocessed env.
#
#     :param env_id: (str) the environment ID
#     :param num_env: (int) the number of environment you wish to have in subprocesses
#     :param seed: (int) the inital seed for RNG
#     :param rank: (int) index of the subprocess
#     :return: (Callable)
#     """
#
#     def _init() -> gym.Env:
#         env = gym.make(env_id)
#         env.seed(seed + rank)
#         return env
#
#     set_random_seed(seed)
#     return _init


env_id = "My_TwoPandaReachPlateJointsDense-v1"
num_cpu = 4
vec_env = make_vec_env(env_id, n_envs=num_cpu)

log_dir = './tensorboard/two_panda_reach_plate_joints_dense_v1/'

total_timesteps = 3500000

# PPO
model = PPO(policy="MultiInputPolicy", env=vec_env, verbose=1, normalize_advantage=True,
            tensorboard_log=log_dir)
model.learn(total_timesteps=total_timesteps)
model.save("./trained/two_panda_reach_plate_joints_dense_v1/two_panda_reach_plate_joints_dense_v1_ppo")
