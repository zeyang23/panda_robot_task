import numpy as np

import sys
sys.path.append("..")

from custom_envs.envs.robots.My_panda import My_Panda
from custom_envs.envs.tasks.My_slide import My_Slide

from panda_gym.envs.core import RobotTaskEnv
from panda_gym.pybullet import PyBullet


class My_PandaSlideEnv(RobotTaskEnv):
    """Slide task wih Panda robot.

    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
    """

    def __init__(self, render: bool = False, reward_type: str = "sparse", control_type: str = "ee") -> None:
        sim = PyBullet(render=render)
        robot = My_Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = My_Slide(sim, reward_type=reward_type)
        super().__init__(robot, task)
