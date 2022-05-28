import gym
import panda_gym
from stable_baselines3 import SAC, HerReplayBuffer

env = gym.make("PandaReach-v2")
log_dir = './tensorboard/panda_reach_v2/'

total_timesteps = 20000

# SAC
model = SAC(policy="MultiInputPolicy", env=env, buffer_size=100000, replay_buffer_class=HerReplayBuffer, verbose=1,
            tensorboard_log=log_dir)
model.learn(total_timesteps=total_timesteps)
model.save("./trained/panda_reach_v2/panda_reach_v2_sac")
