from typing import Any, Dict, Union

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.utils import distance


class My_Reach_Plate(Task):
    def __init__(
            self,
            sim,
            get_ee_position,
            reward_type="sparse",
            distance_threshold=0.15,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position

        self.goal_range_low = np.array([0.4, -0.0, -0.3])
        self.goal_range_high = np.array([0.5, 0.0, -0.2])

        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_sphere(
            body_name="object",
            radius=0.02,
            mass=1.0,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )

        self.sim.create_sphere(
            body_name="target",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )

    def get_obs(self):
        # position, rotation of the object
        object_position = np.array(self.sim.get_base_position("object"))
        object_rotation = np.array(self.sim.get_base_rotation("object"))
        object_velocity = np.array(self.sim.get_base_velocity("object"))
        object_angular_velocity = np.array(self.sim.get_base_angular_velocity("object"))
        observation = np.concatenate(
            [
                object_position,
                object_rotation,
                object_velocity,
                object_angular_velocity,
            ]
        )
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = self.get_ee_position()
        object_position = np.array(self.sim.get_base_position("object"))

        achieved_goal = np.concatenate((ee_position, object_position))
        return achieved_goal

    def reset(self):
        self.goal = self._sample_goal()
        object_position = self._sample_object()
        self.sim.set_base_pose("target", self.goal[:3], np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        goal = self.get_ee_position() + np.array([0.0, 0.0, 0.05])
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        goal += noise
        return np.concatenate((goal, goal))

    def _sample_object(self) -> np.ndarray:
        object_position = self.get_ee_position() + np.array([0.0, 0.0, 0.05])
        return object_position

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=np.float64)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float64)
        else:
            ee_position = self.get_ee_position()
            penalty = distance(achieved_goal[3:], ee_position)

            return -(d + penalty)
