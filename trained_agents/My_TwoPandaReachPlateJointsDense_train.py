import time
import numpy as np

import gym
import panda_gym
import custom_envs

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

env_id = "My_TwoPandaReachPlateJointsDense-v1"
num_cpu = 4
vec_env = make_vec_env(env_id, n_envs=num_cpu)

log_dir = './tensorboard/two_panda_reach_plate_joints_dense_v1/'

checkpoint_callback = CheckpointCallback(save_freq=25000, save_path='model_checkpoints/',
                                         name_prefix='two_reach_plate')

total_timesteps = 6000000

# PPO
model = PPO(policy="MultiInputPolicy", env=vec_env, verbose=1, normalize_advantage=True,
            tensorboard_log=log_dir)
model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
model.save("./trained/two_panda_reach_plate_joints_dense_v1/two_panda_reach_plate_joints_dense_v1_ppo")
