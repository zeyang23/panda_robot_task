import os

import pybullet as p
import pybullet_data as pd

p.connect(p.GUI)
# p.setGravity(0, 0, -10)
p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=0,
                             cameraPitch=-40, cameraTargetPosition=[0.5, -0.9, 0.5])
p.setAdditionalSearchPath(pd.getDataPath())
file_place = os.getcwd()

file_name = file_place + "/custom_envs/envs/robots/panda_plate.urdf"

pandaUid = p.loadURDF(file_name, useFixedBase=True)
tableUid = p.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.65])

targetJointsValue = p.calculateInverseKinematics(pandaUid, 6, [-0.05, 0.0, 0.95], [0.0, 0.0, 0.0, 1.0])

print(targetJointsValue)

for i in range(7):
    p.resetJointState(pandaUid, i, targetJointsValue[i], targetVelocity=0, physicsClientId=0)

while True:
    p.stepSimulation()
