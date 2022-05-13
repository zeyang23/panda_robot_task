import gym
import panda_gym
from stable_baselines3 import SAC, HerReplayBuffer
import custom_envs
from sb3_contrib import TQC

env = gym.make('My_PandaPickAndPlace-v1')

log_dir = './tensorboard/my_panda_slide_v1/'

total_timesteps = 3000000

# SAC
# model = SAC(policy="MultiInputPolicy", env=env, buffer_size=1000000, verbose=1, replay_buffer_class=HerReplayBuffer,
#             tensorboard_log=log_dir)
# model.learn(total_timesteps=total_timesteps)
# model.save("./trained/my_panda_slide_v1/my_panda_slide_v1_sac")

# model = SAC(policy="MultiInputPolicy", env=env, learning_rate=1e-3, buffer_size=1000000, batch_size=2048,
#             replay_buffer_class=HerReplayBuffer, policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2),
#             replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy='future', ), gamma=0.95, tau=0.05,
#             verbose=1,
#             tensorboard_log=log_dir)
# model.learn(total_timesteps=total_timesteps)
# model.save("./trained/my_panda_slide_v1/my_panda_slide_v1_sac")

model = TQC(policy="MultiInputPolicy", env=env, learning_rate=1e-3, buffer_size=1000000, batch_size=2048,
            replay_buffer_class=HerReplayBuffer, policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2),
            replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy='future'), gamma=0.95, tau=0.05,
            verbose=1,
            tensorboard_log=log_dir)
model.learn(total_timesteps=total_timesteps)
model.save("./trained/my_panda_slide_v1/my_panda_slide_v1_tqc")
