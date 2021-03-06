import gym
import panda_gym
from stable_baselines3 import SAC, HerReplayBuffer
import custom_envs
# from sb3_contrib import TQC
from stable_baselines3.common.callbacks import CheckpointCallback

env = gym.make('Two_Obj_PandaPushDense-v1')

log_dir = './tensorboard/two_obj_push_dense_v1/'

total_timesteps = 1500000

# SAC

checkpoint_callback = CheckpointCallback(save_freq=100000, save_path='model_checkpoints/two_obj_push',
                                         name_prefix='two_obj_push')

model = SAC(policy="MultiInputPolicy", env=env, learning_rate=1e-3, buffer_size=1000000, batch_size=2048,
            replay_buffer_class=HerReplayBuffer, policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2),
            replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy='future', ), gamma=0.95, tau=0.05,
            verbose=1,
            tensorboard_log=log_dir)
model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
model.save("./trained/two_obj_push_dense_v1/two_obj_push_dense_v1_sac")
