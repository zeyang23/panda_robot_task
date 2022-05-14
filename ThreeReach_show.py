import gym
import panda_gym
from stable_baselines3 import PPO
import custom_envs

from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("Three_PandaReachDense-v1", render=True)

model = PPO.load("./trained/three_reach_dense_v1/three_reach_dense_v1_ppo", env=env)

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(info)
    env.render()
    if done:
        print('Done')
        obs = env.reset()
