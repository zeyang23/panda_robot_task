import gym
import panda_gym
import custom_envs
import numpy as np

# choose from the commands below

# reach
# env = gym.make('My_PandaReach-v1', render=True)
# env = gym.make('Two_PandaReach-v1', render=True)
# env = gym.make('Three_PandaReach-v1', render=True)
#
# # slide
# env = gym.make('My_PandaSlide-v1', render=True)
#
# # pick and place
# env = gym.make('My_PandaPickAndPlace-v1', render=True)
# env = gym.make('My_TwoPandaPickAndPlace-v1', render=True)
#
# # push
# env = gym.make('Two_PandaPush-v1', render=True)
# env = gym.make('Three_PandaPush-v1', render=True)
#
# env = gym.make('Two_Obj_PandaPush-v1', render=True)
# env = gym.make('Three_Obj_PandaPush-v1', render=True)
#
# # plate
# env = gym.make('My_PandaReachPlate-v1', render=True)
# env = gym.make('My_TwoPandaReachPlate-v1', render=True)
# env.env.sim.physics_client.resetDebugVisualizerCamera(cameraDistance=0.2, cameraYaw=45,
#                                                       cameraPitch=-45, cameraTargetPosition=[1.2, -0.50, 1.5])
#
# # push and pick
env = gym.make('My_PandaPushAndPick-v1', render=True)

obs = env.reset()
done = False
while not done:
    # action = env.action_space.sample()
    action = np.zeros(np.shape(env.action_space))
    obs, reward, done, info = env.step(action)
    print(info)
    env.render()
