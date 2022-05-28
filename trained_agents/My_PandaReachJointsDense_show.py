import gym
import panda_gym
from stable_baselines3 import PPO, SAC
from sb3_contrib import TQC
import custom_envs

env_id = 'My_PandaReachJointsDense'
algorithm_name = 'TQC'

env = gym.make(env_id + '-v1', render=True)

command = algorithm_name + ".load('./trained/' + env_id + '/' + env_id + algorithm_name, env=env)"

model = eval(command)

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(info)
    env.render()
    if done:
        print('Done')
        obs = env.reset()
