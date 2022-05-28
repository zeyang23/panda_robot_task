import gym
import panda_gym
from stable_baselines3 import PPO
import custom_envs

import numpy as np


# TwoReach
# env_id = "Two_PandaReachDense-v1"
# env = gym.make(env_id)
# model = PPO.load("./trained/two_reach_dense_v1/two_reach_dense_v1_ppo", env=env)

# ThreeReach
# env_id = "Three_PandaReachDense-v1"
# env = gym.make(env_id)
# model = PPO.load("./trained/three_reach_dense_v1/three_reach_dense_v1_ppo", env=env)

# PandaReachPlate
# env_id = "My_PandaReachPlateJointsDense-v1"
# env = gym.make(env_id)
# model = PPO.load("./trained/panda_reach_plate_joints_dense_v1/panda_reach_plate_joints_dense_v1_ppo", env=env)

# PandaReachPlate
env_id = "My_TwoPandaReachPlateJointsDense-v1"
env = gym.make(env_id)
model = PPO.load("./trained/two_panda_reach_plate_joints_dense_v1/two_panda_reach_plate_joints_dense_v1_ppo.zip", env=env)

num_episodes = 100

success_indicator = np.zeros([num_episodes, 1])

for i in range(num_episodes):
    obs = env.reset()
    done = False

    while not done:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if info['is_success'] == True:
            success_indicator[i][0] = 1.0

success_rate = np.sum(success_indicator) / num_episodes

print('\n')
print(env_id)
print('success_rate:', success_rate)
