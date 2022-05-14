import gym
import panda_gym
from stable_baselines3 import DDPG, TD3, SAC, PPO, HerReplayBuffer
import custom_envs

env = gym.make('Two_PandaReach-v1')

log_dir = './tensorboard/two_reach_v1/'

total_timesteps = 20000

# # DDPG
# model = DDPG(policy="MultiInputPolicy", env=env, buffer_size=100000, replay_buffer_class=HerReplayBuffer, verbose=1,
#              tensorboard_log=log_dir)
# model.learn(total_timesteps=total_timesteps)
# model.save("./trained/panda_reach_v2/panda_reach_v2_ddpg")

# # TD3
# model = TD3(policy="MultiInputPolicy", env=env, buffer_size=100000, replay_buffer_class=HerReplayBuffer, verbose=1,
#             tensorboard_log=log_dir)
# model.learn(total_timesteps=total_timesteps)
# model.save("./trained/panda_reach_v2/panda_reach_v2_td3")

# # SAC
# model = SAC(policy="MultiInputPolicy", env=env, buffer_size=100000, replay_buffer_class=HerReplayBuffer, verbose=1,
#             tensorboard_log=log_dir)
# model.learn(total_timesteps=total_timesteps)
# model.save("./trained/two_reach_v1/two_reach_v1_sac")

# # SAC
# model = SAC(policy="MultiInputPolicy", env=env, buffer_size=100000, verbose=1,
#             tensorboard_log=log_dir)
# model.learn(total_timesteps=total_timesteps)
# model.save("./trained/two_reach_v1/two_reach_v1_sac")

# PPO
model = PPO(policy="MultiInputPolicy", env=env, verbose=1, normalize_advantage=True,
            tensorboard_log=log_dir)
model.learn(total_timesteps=total_timesteps)
model.save("./trained/two_reach_v1/two_reach_v1_ppo")
