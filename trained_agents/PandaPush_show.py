import gym
import panda_gym
from stable_baselines3 import SAC, HerReplayBuffer

env = gym.make("PandaPush-v2", render=True)

model = SAC.load("./trained/panda_push_v2/panda_push_v2_sac", env=env)

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(info)
    env.render()
    if done:
        print('Done')
        obs = env.reset()
