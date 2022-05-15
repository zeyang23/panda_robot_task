import gym
import panda_gym
import custom_envs

from stable_baselines3 import SAC, HerReplayBuffer

env = gym.make('Three_Obj_PandaPushDense-v1')
log_dir = './tensorboard/three_obj_panda_push_dense_v1/'

# SAC
model = SAC(policy="MultiInputPolicy", env=env, buffer_size=1000000, replay_buffer_class=HerReplayBuffer, verbose=1,
            tensorboard_log=log_dir)

steps_interval = 100000

models_dir = "./trained/three_obj_panda_push_dense_v1"

for i in range(30):
    model.learn(total_timesteps=steps_interval, reset_num_timesteps=False)
    model.save(f"{models_dir}/SAC/{steps_interval * (i + 1)}")
