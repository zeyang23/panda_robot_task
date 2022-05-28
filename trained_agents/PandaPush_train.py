import gym
import panda_gym
from stable_baselines3 import SAC, HerReplayBuffer

env = gym.make("PandaPush-v2")
log_dir = './tensorboard/panda_push_v2/'

total_timesteps = 2000000

# SAC
model = SAC(policy="MultiInputPolicy", env=env, buffer_size=1000000, replay_buffer_class=HerReplayBuffer, verbose=1,
            tensorboard_log=log_dir)
model.learn(total_timesteps=total_timesteps)
model.save("./trained/panda_push_v2/panda_push_v2_sac")
