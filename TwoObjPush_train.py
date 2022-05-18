import gym
import panda_gym
import custom_envs

from stable_baselines3 import PPO, HerReplayBuffer
from sb3_contrib import TQC

from stable_baselines3.common.env_util import make_vec_env

# env_id = "Two_Obj_PandaPushDense-v1"
# num_cpu = 4
# vec_env = make_vec_env(env_id, n_envs=num_cpu)

log_dir = './tensorboard/two_obj_panda_push_v1/'

total_timesteps = 3000000

# PPO
# model = PPO(policy="MultiInputPolicy", env=vec_env, verbose=1, normalize_advantage=True,
#             tensorboard_log=log_dir)
# model.learn(total_timesteps=total_timesteps)
# model.save("./trained/two_obj_panda_push_joints_dense_v1/two_obj_panda_push_joints_dense_v1_ppo")

# model = PPO(policy="MultiInputPolicy", env=vec_env, verbose=1, normalize_advantage=True, batch_size=256, n_steps=512,
#             gamma=0.95, learning_rate=3.56987e-5, ent_coef=0.00238306, clip_range=0.3, n_epochs=5, gae_lambda=0.9,
#             max_grad_norm=2, vf_coef=0.431892,
#             policy_kwargs=dict(log_std_init=-2, ortho_init=False,
#                                net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
#             tensorboard_log=log_dir)
# model.learn(total_timesteps=total_timesteps)
# model.save("./trained/two_obj_panda_push_dense_v1/two_obj_panda_push_dense_v1_ppo")

env = gym.make("Two_Obj_PandaPush-v1")

model = TQC(policy="MultiInputPolicy", env=env, learning_rate=1e-3, buffer_size=1000000, batch_size=2048,
            replay_buffer_class=HerReplayBuffer, policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2),
            replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy='future'), gamma=0.95, tau=0.05,
            verbose=1,
            tensorboard_log=log_dir)
model.learn(total_timesteps=total_timesteps)
model.save("./trained/two_obj_panda_push_v1/two_obj_panda_push_v1_tqc")
