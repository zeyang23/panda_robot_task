import gym
import panda_gym
import custom_envs
from stable_baselines3 import SAC, HerReplayBuffer

env = gym.make('Two_Obj_PandaPushDense-v1', render=True)

model = SAC.load("./trained/two_obj_push_dense_v1/two_obj_push_dense_v1_sac.zip", env=env)

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(info)
    env.render()
    if done:
        print('Done')
        obs = env.reset()
