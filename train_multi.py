import gym
import panda_gym
import custom_envs

from stable_baselines3 import PPO

from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.callbacks import CheckpointCallback

"""
env_list = ['My_PandaReach', 'Two_PandaReach', 'Three_PandaReach',
            'My_PandaSlide',
            'My_PandaPickAndPlace', 'My_TwoPandaPickAndPlace',
            'Two_PandaPush', 'Three_PandaPush',
            'Two_Obj_PandaPush', 'Three_Obj_PandaPush',
            'My_PandaReachPlate', 'My_TwoPandaReachPlate',
            'My_PandaStack']

env_opts = ['Joints', 'Dense']
"""

env_id = 'My_PandaSlideDense'

num_cpu = 4
vec_env = make_vec_env(env_id, n_envs=num_cpu)

log_dir = './tensorboard/' + env_id

checkpoint_callback = CheckpointCallback(save_freq=25000, save_path='model_checkpoints/'+env_id,
                                         name_prefix=env_id)

total_timesteps = 5000000

# PPO
model = PPO(policy="MultiInputPolicy", env=vec_env, verbose=1, normalize_advantage=True,
            tensorboard_log=log_dir)

model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

model.save('./trained/'+env_id+'/'+env_id+model.__class__.__name__)
