import os
import sys

sys.path.append("..")

from gym.envs.registration import register

for reward_type in ["sparse", "dense"]:
    for control_type in ["ee", "joints"]:
        reward_suffix = "Dense" if reward_type == "dense" else ""
        control_suffix = "Joints" if control_type == "joints" else ""
        kwargs = {"reward_type": reward_type, "control_type": control_type}

        register(
            id="My_PandaReach{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="custom_envs.envs:My_PandaReachEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="My_PandaSlide{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="custom_envs.envs:My_PandaSlideEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="My_PandaPickAndPlace{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="custom_envs.envs:My_PandaPickAndPlaceEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="Two_PandaReach{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="custom_envs.envs:Two_PandaReachEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="Three_PandaReach{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="custom_envs.envs:Three_PandaReachEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id="Two_PandaPush{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="custom_envs.envs:Two_PandaPushEnv",
            kwargs=kwargs,
            max_episode_steps=100,
        )

        register(
            id="Two_Obj_PandaPush{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="custom_envs.envs:Two_Obj_PandaPushEnv",
            kwargs=kwargs,
            max_episode_steps=100,
        )

        register(
            id="My_PandaReachPlate{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="custom_envs.envs:My_PandaReachPlateEnv",
            kwargs=kwargs,
            max_episode_steps=100,
        )

        register(
            id="My_TwoPandaReachPlate{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="custom_envs.envs:My_TwoPandaReachPlateEnv",
            kwargs=kwargs,
            max_episode_steps=150,
        )

        register(
            id="My_TwoPandaPickAndPlace{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="custom_envs.envs:My_TwoPandaPickAndPlaceEnv",
            kwargs=kwargs,
            max_episode_steps=100,
        )

        register(
            id="My_PandaPushAndPick{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="custom_envs.envs:My_PandaPushAndPickEnv",
            kwargs=kwargs,
            max_episode_steps=100,
        )
