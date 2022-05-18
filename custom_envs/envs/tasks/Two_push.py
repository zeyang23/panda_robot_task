from typing import Any, Dict, Union

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.utils import distance


class Two_Push(Task):
    def __init__(
            self,
            sim,
            get_ee_position,
            reward_type="sparse",
            distance_threshold=0.05,
            goal_xy_range=0.3,
            obj_xy_range=0.3,
    ) -> None:
        super().__init__(sim)

        self.get_ee_position = get_ee_position

        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.04
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 2, 0])
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])

        self.flag_sub_goal1 = False
        self.flag_sub_goal2 = False

        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_box(
            body_name="object",
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
            body_name="target2",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        object_position = np.array(self.sim.get_base_position("object"))
        object_rotation = np.array(self.sim.get_base_rotation("object"))
        object_velocity = np.array(self.sim.get_base_velocity("object"))
        object_angular_velocity = np.array(self.sim.get_base_angular_velocity("object"))
        observation_object = np.concatenate(
            [
                object_position,
                object_rotation,
                object_velocity,
                object_angular_velocity,
            ]
        )

        observation_subgoal = np.array([self.flag_sub_goal1, self.flag_sub_goal2, 0])

        observation = np.concatenate((observation_subgoal, observation_object))

        return observation

    def get_achieved_goal(self) -> np.ndarray:
        object_position = np.array(self.sim.get_base_position("object"))

        if self.flag_sub_goal1 == False:
            achieved_goal = np.concatenate((object_position, np.array([0.0, 0.0, 0.0])))
        else:
            achieved_goal = np.concatenate((self.sub_goal1, object_position))
        return achieved_goal

    def reset(self) -> None:
        self.flag_sub_goal1 = False
        self.flag_sub_goal2 = False

        self.goal = self._sample_goal()
        object_position = self._sample_object()
        self.sim.set_base_pose("target1", self.sub_goal1, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("target2", self.sub_goal2, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        self.sub_goal1 = np.array([0.0, 0.0, self.object_size / 2])  # z offset for the cube center
        noise1 = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        self.sub_goal1 += noise1

        self.sub_goal2 = np.array([0.0, 0.0, self.object_size / 2])  # z offset for the cube center
        noise2 = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        self.sub_goal2 += noise2

        goal = np.concatenate((self.sub_goal1, self.sub_goal2))

        return goal

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        object_position = np.array([0.0, 0.0, self.object_size / 2])
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        return object_position

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:

        if self.flag_sub_goal1 == False:
            d_sub_goal1 = distance(achieved_goal[:3], desired_goal[:3])
            if d_sub_goal1 < self.distance_threshold:
                self.flag_sub_goal1 = True
        else:
            d = distance(achieved_goal, desired_goal)
            if d < self.distance_threshold:
                self.flag_sub_goal2 = True
            else:
                self.flag_sub_goal2 = False

        return (self.flag_sub_goal1 & self.flag_sub_goal2)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:

        d = distance(achieved_goal, desired_goal)

        if self.reward_type == "sparse":
            reward = 0.0
            if self.is_success(achieved_goal, desired_goal):
                reward = 1.0
            return reward
        else:
            object_position = np.array(self.sim.get_base_position("object"))
            ee_position = np.array(self.get_ee_position())

            ee_obj_distance_threshold = 0.5
            ee_obj_dist = distance(object_position, ee_position)
            penalty = 0.0
            # if ee_obj_dist > ee_obj_distance_threshold:
            #     penalty = ee_obj_dist

            return -(d + penalty)
