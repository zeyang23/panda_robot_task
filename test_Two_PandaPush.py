import gym
import panda_gym
import custom_envs

env = gym.make('Two_PandaPush-v1', render=True)

obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()  # random action
    obs, reward, done, info = env.step(action)
    print(info)
    env.render()  # wait the right amount of time to make the rendering real-time
