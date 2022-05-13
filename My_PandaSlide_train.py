import gym
import panda_gym
from stable_baselines3 import SAC, HerReplayBuffer
import custom_envs

env = gym.make('My_PandaPickAndPlace-v1')

log_dir = './tensorboard/my_panda_slide_v1/'

total_timesteps = 2000000

# SAC
model = SAC(policy="MultiInputPolicy", env=env, buffer_size=1000000, verbose=1, replay_buffer_class=HerReplayBuffer,
            tensorboard_log=log_dir)
model.learn(total_timesteps=total_timesteps)
model.save("./trained/my_panda_slide_v1/my_panda_slide_v1_sac")
