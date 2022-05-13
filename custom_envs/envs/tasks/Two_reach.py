from turtle import end_fill
from typing import Any, Dict, Union

import copy

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.utils import distance


class Two_Reach(Task):
    def __init__(
            self,
            sim,
            get_ee_position,
            reward_type="sparse",
            distance_threshold=0.05,
            goal_range=0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0])
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range])

        self.flag_sub_goal1 = False
        self.flag_sub_goal2 = False

        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_sphere(
            body_name="target1",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )
        self.sim.create_sphere(
            body_name="target2",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )

    def get_obs(self) -> np.ndarray:
        return np.array([])  # no tasak-specific observation

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        return ee_position

    def reset(self) -> None:
        self.goal = self._sample_goal()
        self.sim.set_base_pose("target1", self.sub_goal1, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("target2", self.sub_goal2, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        self.sub_goal1 = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        self.sub_goal2 = self.np_random.uniform(self.goal_range_low, self.goal_range_high)

        goal = copy.deepcopy(self.sub_goal1)
        # goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high)

        return goal

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:

        d = distance(achieved_goal, desired_goal)

        if self.flag_sub_goal1 == False:
            if d < self.distance_threshold:
                self.flag_sub_goal1 = True
                self.goal = copy.deepcopy(self.sub_goal2)
        else:
            if d < self.distance_threshold:
                self.flag_sub_goal2 = True

        return (self.flag_sub_goal1 & self.flag_sub_goal2)
        # return np.array(d < self.distance_threshold, dtype=np.float64)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:

        d = distance(achieved_goal, desired_goal)

        if self.reward_type == "sparse":
            reward = 0.0
            if d < self.distance_threshold:
                reward = 1.0
            return reward
        else:
            return -d
