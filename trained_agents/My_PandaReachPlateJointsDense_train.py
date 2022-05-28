import time
import numpy as np

import gym
import panda_gym
import custom_envs

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

env_id = "My_PandaReachPlateJointsDense-v1"
num_cpu = 4
vec_env = make_vec_env(env_id, n_envs=num_cpu)

log_dir = './tensorboard/panda_reach_plate_joints_dense_v1/'

total_timesteps = 5000000

# PPO
model = PPO(policy="MultiInputPolicy", env=vec_env, verbose=1, normalize_advantage=True,
            tensorboard_log=log_dir)
model.learn(total_timesteps=total_timesteps)
model.save("./trained/panda_reach_plate_joints_dense_v1/panda_reach_plate_joints_dense_v1_ppo")
