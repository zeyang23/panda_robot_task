from typing import Any, Dict, Union

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.utils import distance

from random import choice


class Two_Obj_Push(Task):
    def __init__(
            self,
            sim,
            reward_type="sparse",
            distance_threshold=0.1,
            goal_xy_range=0.2,
            obj_xy_range=0.2,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.04

        self.goal1_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal1_range_high = np.array([-goal_xy_range / 6, -goal_xy_range / 6, 0])

        self.obj1_range_low = np.array([-goal_xy_range / 2, goal_xy_range / 6, 0])
        self.obj1_range_high = np.array([-goal_xy_range / 6, goal_xy_range / 2, 0])

        self.goal2_range_low = np.array([obj_xy_range / 6, -obj_xy_range / 2, 0])
        self.goal2_range_high = np.array([obj_xy_range / 2, -obj_xy_range / 6, 0])

        self.obj2_range_low = np.array([obj_xy_range / 6, obj_xy_range / 6, 0])
        self.obj2_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])

        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_box(
            body_name="object1",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )
        self.sim.create_box(
            body_name="target1",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )
        self.sim.create_box(
            body_name="object2",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([1.0, 0.4, 0.3, 1.0]),
        )
        self.sim.create_box(
            body_name="target2",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([1.0, 0.4, 0.3, 0.3]),
        )

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        object1_position = np.array(self.sim.get_base_position("object1"))
        object1_rotation = np.array(self.sim.get_base_rotation("object1"))
        object1_velocity = np.array(self.sim.get_base_velocity("object1"))
        object1_angular_velocity = np.array(self.sim.get_base_angular_velocity("object1"))
        observation1 = np.concatenate(
            [
                object1_position,
                object1_rotation,
                object1_velocity,
                object1_angular_velocity,
            ]
        )

        object2_position = np.array(self.sim.get_base_position("object2"))
        object2_rotation = np.array(self.sim.get_base_rotation("object2"))
        object2_velocity = np.array(self.sim.get_base_velocity("object2"))
        object2_angular_velocity = np.array(self.sim.get_base_angular_velocity("object2"))
        observation2 = np.concatenate(
            [
                object2_position,
                object2_rotation,
                object2_velocity,
                object2_angular_velocity,
            ]
        )

        observation = np.concatenate((observation1, observation2))

        return observation

    def get_achieved_goal(self) -> np.ndarray:
        object1_position = np.array(self.sim.get_base_position("object1"))
        object2_position = np.array(self.sim.get_base_position("object2"))
        achieved_goal = np.concatenate((object1_position, object2_position))
        return achieved_goal

    def reset(self) -> None:
        # self.goal = self._sample_goal()
        # object_position = self._sample_object()

        self.goal = self._sample_goal()
        object_position = self._sample_object()

        self.sim.set_base_pose("target1", self.goal[:3], np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object1", object_position[:3], np.array([0.0, 0.0, 0.0, 1.0]))

        self.sim.set_base_pose("target2", self.goal[3:], np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object2", object_position[3:], np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal1 = np.array([0.0, 0.0, self.object_size / 2])  # z offset for the cube center
        noise1 = self.np_random.uniform(self.goal1_range_low, self.goal1_range_high)
        goal1 += noise1

        goal2 = np.array([0.0, 0.0, self.object_size / 2])  # z offset for the cube center
        noise2 = self.np_random.uniform(self.goal2_range_low, self.goal2_range_high)
        goal2 += noise2

        goal = np.concatenate((goal1, goal2))

        return goal

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        object1_position = np.array([0.0, 0.0, self.object_size / 2])
        noise1 = self.np_random.uniform(self.obj1_range_low, self.obj1_range_high)
        object1_position += noise1

        object2_position = np.array([0.0, 0.0, self.object_size / 2])
        noise2 = self.np_random.uniform(self.obj2_range_low, self.obj2_range_high)
        object2_position += noise2

        object_position = np.concatenate((object1_position, object2_position))

        return object_position

    def _sample_goal_simple(self) -> np.ndarray:
        """Randomize goal."""
        goal1 = np.array([0.0, 0.0, self.object_size / 2])  # z offset for the cube center
        goal1_x_list = [-0.16, -0.08]
        goal1_y_list = [0.16, 0.08]
        goal1_x = choice(goal1_x_list)
        goal1_y = choice(goal1_y_list)
        noise1 = np.array([goal1_x, goal1_y, 0.0])
        goal1 += noise1

        goal2 = np.array([0.0, 0.0, self.object_size / 2])  # z offset for the cube center
        goal2_x_list = [0.16, 0.08]
        goal2_y_list = [0.16, 0.08]
        goal2_x = choice(goal2_x_list)
        goal2_y = choice(goal2_y_list)
        noise2 = np.array([goal1_x, goal1_y, 0.0])
        goal2 += noise2

        goal = np.concatenate((goal1, goal2))

        return goal

    def _sample_object_simple(self) -> np.ndarray:
        """Randomize start position of object."""
        object1_position = np.array([0.0, 0.0, self.object_size / 2])
        obj1_x_list = [-0.16, -0.08]
        obj1_y_list = [-0.16, -0.08]
        obj1_x = choice(obj1_x_list)
        obj1_y = choice(obj1_y_list)
        noise1 = np.array([obj1_x, obj1_y, 0.0])
        object1_position += noise1

        object2_position = np.array([0.0, 0.0, self.object_size / 2])
        obj2_x_list = [0.16, 0.08]
        obj2_y_list = [-0.16, -0.08]
        obj2_x = choice(obj2_x_list)
        obj2_y = choice(obj2_y_list)
        noise2 = np.array([obj2_x, obj2_y, 0.0])
        object2_position += noise2

        object_position = np.concatenate((object1_position, object2_position))

        return object_position

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=np.float64)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float64)
        else:
            return -d
