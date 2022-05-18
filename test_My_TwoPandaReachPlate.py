import gym
import numpy as np
import panda_gym
import custom_envs

env = gym.make('My_TwoPandaReachPlateJoints-v1', render=True)

obs = env.reset()

env.env.sim.physics_client.resetDebugVisualizerCamera(cameraDistance=0.2, cameraYaw=45,
                                                      cameraPitch=-45, cameraTargetPosition=[1.2, -0.50, 1.5])
done = False
while True:
    # action = env.action_space.sample()  # random action
    action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    obs, reward, done, info = env.step(action)
    print(info)
    env.render()  # wait the right amount of time to make the rendering real-time
