import gym
import panda_gym
from sb3_contrib import TQC
import custom_envs

env = gym.make("My_PandaSlideDense-v1", render=True)

model = TQC.load("./trained/my_panda_slide_dense_v1/my_panda_slide_dense_v1_tqc", env=env)

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(info)
    env.render()
    if done:
        print('Done')
        obs = env.reset()
