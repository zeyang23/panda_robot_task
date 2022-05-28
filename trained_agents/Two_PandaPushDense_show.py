import gym
import panda_gym
import custom_envs
from stable_baselines3 import SAC, HerReplayBuffer
from sb3_contrib import TQC

env = gym.make('Two_PandaPushDense-v1', render=True)

model = TQC.load("./trained/two_push_dense_v1/two_push_dense_v1_sac.zip", env=env)

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(info)
    env.render()
    if done:
        print('Done')
        obs = env.reset()
