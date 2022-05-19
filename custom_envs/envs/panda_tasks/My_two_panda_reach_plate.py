import numpy as np

import sys

sys.path.append("..")

from custom_envs.envs.robots.My_panda_plate_new import My_Panda_Plate_New
from custom_envs.envs.tasks.My_two_reach_plate import My_Two_Reach_Plate

from panda_gym.envs.core import RobotTaskEnv
from panda_gym.pybullet import PyBullet


class My_TwoPandaReachPlateEnv(RobotTaskEnv):
    """Reach task wih Panda robot.

    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
    """

    def __init__(self, render: bool = False, reward_type: str = "sparse", control_type: str = "ee") -> None:
        sim = PyBullet(render=render)
        robot = My_Panda_Plate_New(sim, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = My_Two_Reach_Plate(sim, reward_type=reward_type, get_ee_position=robot.get_ee_position)
        super().__init__(robot, task)
