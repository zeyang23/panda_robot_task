import gym
import panda_gym
from stable_baselines3 import SAC, HerReplayBuffer
import custom_envs
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import CheckpointCallback

env_id = 'My_PandaReachDense'

env = gym.make(env_id+'-v1')

log_dir = './tensorboard/' + env_id

total_timesteps = 2000000

checkpoint_callback = CheckpointCallback(save_freq=100000, save_path='model_checkpoints/'+env_id,
                                         name_prefix=env_id)

model = TQC(policy="MultiInputPolicy", env=env, learning_rate=1e-3, buffer_size=1000000, batch_size=2048,
            replay_buffer_class=HerReplayBuffer, policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2),
            replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy='future', ), gamma=0.95, tau=0.05,
            verbose=1,
            tensorboard_log=log_dir)

model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

model.save('./trained/'+env_id+'/'+env_id+model.__class__.__name__)
