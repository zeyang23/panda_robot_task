from typing import Any, Dict, Tuple, Union

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.utils import distance

from random import choice


class My_Stack(Task):
    def __init__(
            self,
            sim,
            reward_type="sparse",
            distance_threshold=0.1,
            goal_xy_range=0.3,
            obj_xy_range=0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold

        self.flag_sub_goal1 = False
        self.flag_sub_goal2 = False

        self.object_size = 0.04
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 2, 0])
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_box(
            body_name="object1",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=2.0,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.1, 0.9, 1.0]),
        )
        self.sim.create_box(
            body_name="target1",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, 0.05]),
            rgba_color=np.array([0.1, 0.1, 0.9, 0.3]),
        )
        self.sim.create_box(
            body_name="object2",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.5, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )
        self.sim.create_box(
            body_name="target2",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.5, 0.0, 0.05]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        object1_position = np.array(self.sim.get_base_position("object1"))
        object1_rotation = np.array(self.sim.get_base_rotation("object1"))
        object1_velocity = np.array(self.sim.get_base_velocity("object1"))
        object1_angular_velocity = np.array(self.sim.get_base_angular_velocity("object1"))
        object2_position = np.array(self.sim.get_base_position("object2"))
        object2_rotation = np.array(self.sim.get_base_rotation("object2"))
        object2_velocity = np.array(self.sim.get_base_velocity("object2"))
        object2_angular_velocity = np.array(self.sim.get_base_angular_velocity("object2"))
        observation_object = np.concatenate(
            [
                object1_position,
                object1_rotation,
                object1_velocity,
                object1_angular_velocity,
                object2_position,
                object2_rotation,
                object2_velocity,
                object2_angular_velocity,
            ]
        )

        observation_subgoal = np.array([self.flag_sub_goal1, self.flag_sub_goal2])

        observation = np.concatenate((observation_subgoal, observation_object))

        return observation

    def get_achieved_goal(self) -> np.ndarray:
        object1_position = self.sim.get_base_position("object1")
        object2_position = self.sim.get_base_position("object2")
        # achieved_goal = np.concatenate((object1_position, object2_position))

        distance_subgoal1 = distance(object1_position, self.sub_goal1)

        if distance_subgoal1 <= 0.05:
            self.flag_sub_goal1 = True
            achieved_goal = np.concatenate((self.sub_goal1, object2_position))
        else:
            self.flag_sub_goal1 = False
            achieved_goal = np.concatenate((object1_position, np.array([0.0, 0.0, 0.0])))

        return achieved_goal

    def reset(self) -> None:
        self.flag_sub_goal1 = False
        self.flag_sub_goal2 = False

        self.goal = self._sample_goal_simple()
        object1, object2 = self._sample_object_simple()
        self.sim.set_base_pose("target1", self.goal[:3], np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("target2", self.goal[3:], np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object1", object1, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object2", object2, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        goal1 = np.array([0.0, 0.0, self.object_size / 2])  # z offset for the cube center
        goal2 = np.array([0.0, 0.0, 3 * self.object_size / 2])  # z offset for the cube center
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        goal1 += noise
        goal2 += noise
        return np.concatenate((goal1, goal2))

    def _sample_objects(self) -> Tuple[np.ndarray, np.ndarray]:
        # while True:  # make sure that cubes are distant enough
        object1_position = np.array([0.0, 0.0, self.object_size / 2])
        object2_position = np.array([0.0, 0.0, 3 * self.object_size / 2])
        noise1 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        noise2 = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object1_position += noise1
        object2_position += noise2
        # if distance(object1_position, object2_position) > 0.1:
        return object1_position, object2_position

    def _sample_goal_simple(self) -> np.ndarray:
        """Randomize goal."""
        self.sub_goal1 = np.array([0.0, 0.0, self.object_size / 2])  # z offset for the cube center
        goal1_x_list = [-0.05, 0, 0.05]
        goal1_x = choice(goal1_x_list)
        goal1_y = 0.0

        noise1 = np.array([goal1_x, goal1_y, 0.0])

        self.sub_goal1 += noise1

        self.sub_goal2 = np.array([0.0, 0.0, 3 * self.object_size / 2])  # z offset for the cube center

        self.sub_goal2 += noise1

        goal = np.concatenate((self.sub_goal1, self.sub_goal2))

        return goal

    def _sample_object_simple(self) -> np.ndarray:
        object1 = np.array([0.0, 0.0, self.object_size / 2])  # z offset for the cube center
        object1_x_list = [-0.1, 0, 0.1]
        object1_x = choice(object1_x_list)
        object1_y = -0.15
        noise1 = np.array([object1_x, object1_y, 0.0])
        object1 += noise1

        object2 = np.array([0.0, 0.0, self.object_size / 2])  # z offset for the cube center
        object2_x_list = [-0.1, 0, 0.1]
        object2_x = choice(object2_x_list)
        object2_y = 0.15
        noise2 = np.array([object2_x, object2_y, 0.0])
        object2 += noise2

        return object1, object2

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        # must be vectorized !!

        object1_position = self.sim.get_base_position("object1")
        object2_position = self.sim.get_base_position("object2")
        # achieved_goal = np.concatenate((object1_position, object2_position))

        distance_subgoal1 = distance(object1_position, self.sub_goal1)
        distance_subgoal2 = distance(object2_position, self.sub_goal2)

        if distance_subgoal1 <= 0.05:
            self.flag_sub_goal1 = True
            if distance_subgoal2 <= 0.05:
                self.flag_sub_goal2 = True
            else:
                self.flag_sub_goal2 = False
        else:
            self.flag_sub_goal1 = False
            self.flag_sub_goal2 = False

        return (self.flag_sub_goal1 & self.flag_sub_goal2)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array((d > self.distance_threshold), dtype=np.float64)
        else:
            return -d
