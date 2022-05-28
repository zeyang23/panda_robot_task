import gym
import panda_gym
from stable_baselines3 import SAC, HerReplayBuffer

env = gym.make("PandaPickAndPlace-v2")
log_dir = './tensorboard/panda_pick_and_place_v2/'

total_timesteps = 6000000

# SAC
model = SAC(policy="MultiInputPolicy", env=env, learning_rate=1e-3, buffer_size=1000000, batch_size=2048,
            replay_buffer_class=HerReplayBuffer, policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2),
            replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy='future', ), gamma=0.95, tau=0.05,
            verbose=1,
            tensorboard_log=log_dir)
model.learn(total_timesteps=total_timesteps)
model.save("./trained/panda_pick_and_place_v2/panda_pick_and_place_v2_sac")
