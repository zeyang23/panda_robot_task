import gym
import panda_gym
from stable_baselines3 import PPO
from sb3_contrib import TQC
import custom_envs

env = gym.make("My_TwoPandaReachPlateJointsDense-v1", render=True)

model = PPO.load("./trained/two_panda_reach_plate_joints_dense_v1/two_panda_reach_plate_joints_dense_v1_ppo", env=env)

obs = env.reset()

env.env.sim.physics_client.resetDebugVisualizerCamera(cameraDistance=0.2, cameraYaw=45,
                                                      cameraPitch=-45, cameraTargetPosition=[1.2, -0.50, 1.5])

for i in range(1000):
    # action, _state = model.predict(obs, deterministic=True)
    action, _state = model.predict(obs)
    obs, reward, done, info = env.step(action)
    print(info)
    env.render()
    if done:
        print('Done')
        obs = env.reset()
