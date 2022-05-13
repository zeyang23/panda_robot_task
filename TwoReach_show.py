import gym
import panda_gym
from stable_baselines3 import SAC, HerReplayBuffer
import custom_envs

env = gym.make("Two_PandaReach-v1", render=True)

model = SAC.load("./trained/two_reach_v1/two_reach_v1_sac", env=env)

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(info)
    env.render()
    if done:
        print('Done')
        obs = env.reset()
