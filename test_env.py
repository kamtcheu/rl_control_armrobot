import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.utils import EzPickle

# class NewStyleRobotEnv(MujocoEnv, EzPickle):
#     # 1. Define standard Gymnasium rendering formats
#     metadata = {
#         "render_modes": ["human", "rgb_array", "depth_array"],
#         "render_fps": 50,
#     }

#     def __init__(self, xml_file="robot_scene.xml", frame_skip=5, **kwargs):
#         # 2. Correct constructor handling using EzPickle for serialization
#         EzPickle.__init__(self, xml_file, frame_skip, **kwargs)
        
#         # 3. Pass full path or package-relative assets directly to MujocoEnv
#         # In v5+, default_camera_config can be customized via a dict
#         super().__init__(
#             model_path=xml_file,
#             frame_skip=frame_skip,
#             observation_space=None,  # Automatically inferred if generated below
#             default_camera_config={"distance": 3.0, "elevation": -20.0},
#             **kwargs
#         )

#     def step(self, action):
#         # 4. Step the simulation forward using frame skipping
#         self.do_simulation(action, self.frame_skip)
        
#         obs = self._get_obs()
        
#         # 5. Handle standard 5-tuple returns (Separated Terminal vs Truncated)
#         reward = self._compute_reward(action)
#         terminated = self._check_termination()
#         truncated = False
#         info = {"x_position": self.data.qpos[0]}
        
#         return obs, reward, terminated, truncated, info

#     def reset_model(self):
#         # 6. Replaces old model reset pipelines; defines state randomization
#         noise_low, noise_high = -0.01, 0.01
#         qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
#         qvel = self.init_qvel + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv)
        
#         # Directly manipulate underlying C structures
#         self.set_state(qpos, qvel)
#         return self._get_obs()

#     def _get_obs(self):
#         # 7. Fast, non-copy pointer slicing directly from dynamic simulation memory
#         return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])

#     def _compute_reward(self, action):
#         return 1.0  # Custom reward function logic here
        
#     def _check_termination(self):
#         return bool(self.data.qpos[2] < 0.2)  # Example: Terminate if the robot falls

#%%

# import gymnasium as gym
# from stable_baselines3 import SAC # Change this if you used SAC, TD3, A2C, etc.
# from trajectory_env4 import ArmRobot
# import imageio
# import os
# os.environ["DISPLAY"] = 0
# # 1. Configuration
# # Replace these with your specific environment and model file
# env_name = "HalfCheetah-v4" 
# model_path = "sac_eval_checkpoints/ev4/sac_eval40_500000_steps" 

# # 2. Initialize the environment with human rendering
# # render_mode="human" automatically opens a window and renders each step
# # env = ArmRobot(null_threshold=1e-8, work_box_size=0.1, threshold_pos=50e-6, threshold_ori=1.50, start_target_idx=0, max_joint_delta = 3.0, render_mode="rgb_array", site_name="needle_holder_site", episode_len=250, num_points=20)
# #env = gym.make(env_name, render_mode="human")
# gym.register(id="my_master_envs/ArmRobot-v4", entry_point="trajectory_env4:ArmRobot")
# env = gym.make("my_master_envs/ArmRobot-v4", null_threshold=1e-8, work_box_size=0.1, threshold_pos=50e-6, threshold_ori=1.50, start_target_idx=0, max_joint_delta = 3.0, render_mode="human", site_name="needle_holder_site", episode_len=250, num_points=20)
# # 3. Load the trained model
# # Note: Ensure you import the exact algorithm class you used for training
# print(f"Loading model from {model_path}...")
# model = SAC.load(model_path, env=env)

# # 4. Evaluation Loop
# num_episodes = 5
# #frames = []
# for ep in range(num_episodes):
#     # Reset the environment at the start of each episode
#     obs, info = env.reset()
#     done = False
#     truncated = False
#     episode_reward = 0.0
    
#     print(f"Starting Episode {ep + 1}...")

#     # Run the episode until the agent dies (done) or hits the time limit (truncated)
#     while not (done or truncated):
#         # Predict the action to take. 
#         # deterministic=True is usually best for testing/evaluating trained agents.
#         action, _states = model.predict(obs, deterministic=True)
        
#         # Take a step in the environment
#         obs, reward, done, truncated, info = env.step(action)
#         episode_reward += reward

        
#         # frames.append(env.render())
        
#     print(f"Episode {ep + 1} finished with Total Reward: {episode_reward:.2f}")
# # print("Saving test_env4.mp4...")
# # imageio.mimsave("test_env4.mp4", frames, fps=30)
# # print("Done! You can now watch test_env4.mp4")
# # 5. Clean up
# env.close()
# print("Testing complete.")
# %%
import gymnasium as gym
import imageio
from stable_baselines3 import SAC 
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from theEnv import ArmRobot
# os.environ["DISPLAY"] = "0"
# # os.environ["MUJOCO_GL"] = "egl"
model_path = "sac_eval_checkpoints/nolook_envOne_sac_eval_1000000_steps" 

# 2. Initialize the environment with human rendering
# render_mode="human" automatically opens a window and renders each step
env = ArmRobot(render_mode="human",episode_len=200)#null_threshold=1e-8, work_box_size=0.1, threshold_pos=50e-6, threshold_ori=1.5, start_target_idx=0, max_joint_delta = 3.0, render_mode=None, site_name="needle_holder_site", episode_len=250, num_points=20)
#env = gym.make(env_name, render_mode="human")
# gym.register(id="my_master_envs/ArmRobot-v4", entry_point="trajectory_env4:ArmRobot")
# env = gym.make("my_master_envs/ArmRobot-v4", null_threshold=1e-8, work_box_size=0.1, threshold_pos=50e-6, threshold_ori=1.50, start_target_idx=0, max_joint_delta = 0.5, render_mode=None, site_name="needle_holder_site", episode_len=250, num_points=20)
#env = gym.wrappers.RecordVideo(env=env, video_folder="./saved_models")
# 3. Load the trained model
#env.reset()
# Note: Ensure you import the exact algorithm class you used for training
print(f"Loading model from {model_path}...")
model = SAC.load(os.path.abspath(model_path), env=env)

# 4. Evaluation Loop
num_episodes = 10
positions = []
orientations = []
frames = []
for ep in range(num_episodes):
    # Reset the environment at the start of each episode
    obs, info = env.reset()
    done = False
    truncated = False
    episode_reward = 0.0
    positions = []
    orientations = []
    frames = []
    print(f"Starting Episode {ep + 1}...")
    positions.append(info["current_pos"])
    orientations.append(info["current_ori"])

    # Run the episode until the agent dies (done) or hits the time limit (truncated)
    while not (done or truncated):
        # Predict the action to take. 
        # deterministic=True is usually best for testing/evaluating trained agents.
        action, _states = model.predict(obs, deterministic=True)
        print(action)
        
        # Take a step in the environment
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        positions.append(info["current_pos"])
        orientations.append(info["current_ori"])
        
        #frames.append(env.render())
    import matplotlib.pyplot as plt
    import numpy as np
    fig = plt.figure(figsize=(18, 6))
    #positions = #[np.array([ 240.18, -243.96,   66.58]), np.array([ 241.03, -244.4 ,   67.4 ]), np.array([ 251.54, -242.12,   74.57]), np.array([ 251.49, -242.08,   74.89]), np.array([ 245.88, -240.06,   67.53]), np.array([ 246.56, -241.35,   68.41]), np.array([ 249.17, -242.81,   72.98]), np.array([ 247.45, -243.74,   71.88]), np.array([ 246.3 , -245.03,   70.76]), np.array([ 246.93, -245.6 ,   70.4 ]), np.array([ 246.98, -245.96,   70.02]), np.array([ 246.99, -245.98,   70.01]), np.array([ 247.27, -246.77,   70.03]), np.array([ 247.17, -246.84,   70.08]), np.array([ 247.15, -246.87,   70.01]), np.array([ 247.16, -246.84,   70.04]), np.array([ 247.17, -246.84,   70.01]), np.array([ 247.18, -246.82,   70.03]), np.array([ 247.16, -246.82,   70.02]), np.array([ 247.13, -246.79,   70.  ]), np.array([ 247.13, -246.78,   69.99]), np.array([ 247.14, -246.77,   70.  ]), np.array([ 247.14, -246.76,   70.  ]), np.array([ 247.17, -246.77,   70.02]), np.array([ 247.16, -246.75,   70.01]), np.array([ 247.18, -246.76,   70.  ]), np.array([ 247.16, -246.76,   69.99]), np.array([ 247.13, -246.75,   69.98]), np.array([ 247.13, -246.74,   69.95]), np.array([ 247.15, -246.78,   69.96]), np.array([ 247.18, -246.78,   70.  ]), np.array([ 247.17, -246.79,   70.  ]), np.array([ 247.17, -246.81,   69.99]), np.array([ 247.17, -246.8 ,   69.99]), np.array([ 247.66, -247.45,   69.97]), np.array([ 247.59, -247.46,   70.03]), np.array([ 247.62, -247.48,   70.02]), np.array([ 247.63, -247.46,   70.04]), np.array([ 247.63, -247.47,   70.03]), np.array([ 247.64, -247.46,   70.04]), np.array([ 247.66, -247.47,   70.03]), np.array([ 247.66, -247.48,   70.03]), np.array([ 247.66, -247.46,   70.03]), np.array([ 247.64, -247.47,   70.02]), np.array([ 247.65, -247.45,   70.  ]), np.array([ 247.63, -247.46,   69.98]), np.array([ 247.63, -247.48,   70.  ]), np.array([ 247.65, -247.48,   70.  ]), np.array([ 247.66, -247.5 ,   69.99]), np.array([ 247.64, -247.49,   69.98]), np.array([ 248.29, -248.02,   70.03]), np.array([ 248.27, -248.  ,   70.05]), np.array([ 248.27, -248.  ,   70.05]), np.array([ 248.29, -248.  ,   70.05]), np.array([ 248.28, -248.  ,   70.04]), np.array([ 248.27, -248.  ,   70.03]), np.array([ 248.28, -247.98,   70.01]), np.array([ 248.28, -248.  ,   70.01]), np.array([ 248.27, -248.  ,   70.  ]), np.array([ 248.28, -248.  ,   70.  ]), np.array([ 248.25, -248.01,   69.99]), np.array([ 248.95, -248.37,   70.01]), np.array([ 248.92, -248.38,   70.02]), np.array([ 248.94, -248.36,   70.02]), np.array([ 248.93, -248.36,   70.01]), np.array([ 248.95, -248.38,   70.01]), np.array([ 248.94, -248.36,   70.01]), np.array([ 248.93, -248.38,   70.  ]), np.array([ 248.93, -248.38,   70.  ]), np.array([ 249.56, -248.64,   69.96]), np.array([ 249.58, -248.65,   69.97]), np.array([ 249.58, -248.62,   70.01]), np.array([ 249.58, -248.61,   70.01]), np.array([ 249.57, -248.62,   70.  ]), np.array([ 250.19, -248.79,   69.98]), np.array([ 250.19, -248.81,   69.99]), np.array([ 250.17, -248.78,   70.  ]), np.array([ 250.17, -248.79,   70.  ]), np.array([ 250.18, -248.78,   70.  ]), np.array([ 250.75, -248.9 ,   70.  ]), np.array([ 250.73, -248.89,   70.  ]), np.array([ 250.73, -248.89,   70.  ]), np.array([ 250.71, -248.88,   70.  ]), np.array([ 251.25, -248.94,   70.03]), np.array([ 251.24, -248.96,   70.  ]), np.array([ 251.22, -248.99,   69.98]), np.array([ 251.77, -248.98,   70.02]), np.array([ 251.78, -249.01,   70.  ]), np.array([ 251.74, -248.99,   70.01]), np.array([ 252.25, -248.98,   70.  ]), np.array([ 252.25, -248.99,   70.01]), np.array([ 252.25, -248.97,   70.  ]), np.array([ 252.76, -248.95,   70.01]), np.array([ 252.76, -248.95,   70.  ]), np.array([ 252.75, -248.96,   70.  ]), np.array([ 253.28, -248.88,   70.  ]), np.array([ 253.27, -248.89,   70.  ]), np.array([ 253.27, -248.88,   70.  ]), np.array([ 253.25, -248.88,   70.  ]), np.array([ 253.81, -248.81,   70.  ]), np.array([ 253.81, -248.77,   70.  ]), np.array([ 253.83, -248.77,   70.  ]), np.array([ 253.8 , -248.77,   70.01]), np.array([ 254.41, -248.67,   69.99]), np.array([ 254.41, -248.61,   70.  ]), np.array([ 254.42, -248.62,   70.01]), np.array([ 254.4 , -248.62,   69.99]), np.array([ 255.04, -248.4 ,   69.96]), np.array([ 255.03, -248.38,   69.97]), np.array([ 255.05, -248.38,   69.97]), np.array([ 255.06, -248.37,   70.  ]), np.array([ 255.05, -248.38,   70.  ]), np.array([ 255.1 , -248.4 ,   69.99]), np.array([ 255.69, -248.02,   70.02]), np.array([ 255.66, -248.  ,   70.  ]), np.array([ 255.68, -248.  ,   69.99]), np.array([ 255.68, -248.01,   69.99]), np.array([ 255.71, -247.99,   69.99]), np.array([ 255.71, -248.  ,   70.01]), np.array([ 255.7 , -247.99,   69.99]), np.array([ 255.73, -248.01,   70.02]), np.array([ 256.32, -247.5 ,   70.03]), np.array([ 256.3 , -247.5 ,   70.02]), np.array([ 256.3 , -247.49,   70.02]), np.array([ 256.3 , -247.49,   70.01]), np.array([ 256.31, -247.48,   69.99]), np.array([ 256.31, -247.48,   70.  ]), np.array([ 256.32, -247.46,   70.  ]), np.array([ 256.33, -247.48,   70.  ]), np.array([ 256.34, -247.46,   70.01]), np.array([ 256.34, -247.48,   70.  ]), np.array([ 256.33, -247.47,   69.99]), np.array([ 256.53, -247.39,   70.06]), np.array([ 256.82, -246.83,   70.03]), np.array([ 256.82, -246.82,   70.02]), np.array([ 256.82, -246.81,   70.03]), np.array([ 256.82, -246.82,   70.02]), np.array([ 256.81, -246.82,   70.02]), np.array([ 256.82, -246.82,   70.02]), np.array([ 256.81, -246.82,   70.01]), np.array([ 256.82, -246.8 ,   70.02]), np.array([ 256.83, -246.8 ,   70.  ]), np.array([ 256.83, -246.79,   70.01]), np.array([ 256.83, -246.78,   70.01]), np.array([ 256.83, -246.78,   70.01]), np.array([ 256.84, -246.78,   70.01]), np.array([ 256.85, -246.79,   70.01]), np.array([ 256.85, -246.81,   70.01]), np.array([ 256.8 , -246.77,   69.99]), np.array([ 257.07, -246.69,   70.05]), np.array([ 256.82, -246.79,   70.  ]), np.array([ 256.83, -246.79,   69.92]), np.array([ 256.79, -246.77,   70.  ]), np.array([ 256.84, -246.79,   69.94]), np.array([ 257.04, -245.96,   70.03]), np.array([ 257.04, -245.99,   70.03]), np.array([ 257.03, -246.01,   70.04]), np.array([ 257.02, -246.  ,   70.02]), np.array([ 257.03, -246.  ,   70.03]), np.array([ 257.03, -246.  ,   70.03]), np.array([ 257.02, -246.  ,   70.02]), np.array([ 257.02, -246.01,   70.02]), np.array([ 257.02, -246.01,   70.02]), np.array([ 257.02, -246.01,   70.01]), np.array([ 257.02, -246.01,   70.02]), np.array([ 257.03, -246.  ,   70.01]), np.array([ 257.04, -246.  ,   70.01]), np.array([ 257.04, -245.99,   70.01]), np.array([ 257.04, -245.99,   70.02]), np.array([ 257.03, -245.98,   70.02]), np.array([ 257.04, -245.98,   70.02]), np.array([ 257.05, -245.99,   70.02]), np.array([ 257.05, -245.99,   70.02]), np.array([ 257.05, -246.03,   70.02]), np.array([ 257.06, -246.05,   70.02]), np.array([ 257.05, -246.06,   70.03]), np.array([ 257.05, -246.07,   70.04]), np.array([ 257.04, -246.05,   70.04]), np.array([ 257.05, -246.05,   70.03]), np.array([ 257.05, -246.05,   70.03]), np.array([ 257.05, -246.06,   70.04]), np.array([ 257.04, -246.07,   70.03]), np.array([ 257.05, -246.06,   70.04]), np.array([ 257.04, -246.07,   70.04]), np.array([ 257.03, -246.07,   70.03]), np.array([ 257.04, -246.08,   70.03]), np.array([ 257.03, -246.08,   70.04]), np.array([ 257.03, -246.09,   70.03]), np.array([ 257.03, -246.09,   70.04]), np.array([ 257.02, -246.1 ,   70.04]), np.array([ 257.03, -246.1 ,   70.03]), np.array([ 257.02, -246.11,   70.04]), np.array([ 257.02, -246.12,   70.04]), np.array([ 257.02, -246.12,   70.05]), np.array([ 257.02, -246.12,   70.04]), np.array([ 257.01, -246.12,   70.04]), np.array([ 257.02, -246.12,   70.05]), np.array([ 257.03, -246.12,   70.04]), np.array([ 257.02, -246.13,   70.05]), np.array([ 257.02, -246.12,   70.05]), np.array([ 257.03, -246.13,   70.05]), np.array([ 257.02, -246.13,   70.05]), np.array([ 257.03, -246.13,   70.05]), np.array([ 257.02, -246.13,   70.05]), np.array([ 257.03, -246.13,   70.05]), np.array([ 257.03, -246.13,   70.05]), np.array([ 257.03, -246.13,   70.06]), np.array([ 257.03, -246.13,   70.05]), np.array([ 257.03, -246.13,   70.06]), np.array([ 257.03, -246.13,   70.05]), np.array([ 257.03, -246.13,   70.06]), np.array([ 257.04, -246.13,   70.05]), np.array([ 257.04, -246.13,   70.06]), np.array([ 257.04, -246.14,   70.06]), np.array([ 257.03, -246.13,   70.06]), np.array([ 257.04, -246.13,   70.06]), np.array([ 257.04, -246.13,   70.06]), np.array([ 257.05, -246.13,   70.06]), np.array([ 257.04, -246.13,   70.06]), np.array([ 257.04, -246.13,   70.07]), np.array([ 257.05, -246.13,   70.06]), np.array([ 257.04, -246.12,   70.06]), np.array([ 257.05, -246.13,   70.07]), np.array([ 257.05, -246.13,   70.06]), np.array([ 257.05, -246.12,   70.06]), np.array([ 257.04, -246.13,   70.07]), np.array([ 257.06, -246.13,   70.06]), np.array([ 257.05, -246.12,   70.07]), np.array([ 257.05, -246.13,   70.07]), np.array([ 257.05, -246.12,   70.06]), np.array([ 257.05, -246.12,   70.07]), np.array([ 257.05, -246.13,   70.07]), np.array([ 257.06, -246.12,   70.07]), np.array([ 257.05, -246.13,   70.07]), np.array([ 257.06, -246.12,   70.07]), np.array([ 257.05, -246.12,   70.07]), np.array([ 257.06, -246.12,   70.07]), np.array([ 257.06, -246.13,   70.07]), np.array([ 257.05, -246.12,   70.07]), np.array([ 257.06, -246.12,   70.07]), np.array([ 257.06, -246.12,   70.08]), np.array([ 257.05, -246.12,   70.07]), np.array([ 257.06, -246.12,   70.07]), np.array([ 257.06, -246.12,   70.08]), np.array([ 257.06, -246.12,   70.07]), np.array([ 257.06, -246.12,   70.08]), np.array([ 257.06, -246.12,   70.07]), np.array([ 257.06, -246.12,   70.08]), np.array([ 257.06, -246.12,   70.07]), np.array([ 257.07, -246.12,   70.08]), np.array([ 257.06, -246.12,   70.08]), np.array([ 257.06, -246.11,   70.08]), np.array([ 257.07, -246.12,   70.07]), np.array([ 257.05, -246.12,   70.08]), np.array([ 257.07, -246.12,   70.08]), np.array([ 257.06, -246.11,   70.08]), np.array([ 257.07, -246.12,   70.08]), np.array([ 257.06, -246.12,   70.08]), np.array([ 247.16, -250.63,   72.63]), np.array([ 246.76, -249.12,   71.56])]
    positions = np.array([a.tolist() for a in positions])
    O_t = np.array([orientation[:, 0] for orientation in orientations]).T
    # Subplot 1: Target Needle Tip
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    ax1.plot3D(positions[:, 0], positions[:, 1], positions[:, 2], 'b', linewidth=2)
    ax1.scatter3D(positions[:, 0], positions[:, 1], positions[:, 2], 'b', linewidth=2)
    ax1.plot3D(env.target_path[0, :], env.target_path[1,: ], env.target_path[ 2,:], 'g', linewidth=2)
    ax1.scatter3D(env.target_path[0, :], env.target_path[1,: ], env.target_path[ 2,:], 'g', linewidth=2)
    #ax1.quiver(positions[:, 0], positions[:, 1], positions[:, 2], O_t[0, :], O_t[1, :], O_t[2, :], color='r', length=0.3)
    ax1.set_title('Target Needle Tip\n(global pos + global orient)')
    ax1.set_xlabel('x_t')
    ax1.set_ylabel('y_t')
    ax1.set_zlabel('z_t')
    ax1.set_xlim(247, 257)
    ax1.set_ylim(-253, -244)
    ax1.set_zlim(70.0, 70.4)

    ax1.set_box_aspect((10, 4, 0.4))
    ax1.view_init(elev=20, azim=-80)
    #ax1.view_init(elev=90, azim=0) # Emulates view(90,0)
    ax1.grid(True)

    # Subplot 2: Target TCP
    # ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    # ax2.plot3D(P_tcp[0, :], P_tcp[1, :], P_tcp[2, :], 'b', linewidth=2)
    # ax2.quiver(P_tcp[0, :], P_tcp[1, :], P_tcp[2, :], O_tcp[0, :], O_tcp[1, :], O_tcp[2, :], color='r', length=0.2)
    # ax2.set_title('Target tcp\n(global position)')
    # ax2.set_xlabel('x_tcp')
    # ax2.set_ylabel('y_tcp')
    # ax2.set_zlabel('z_tcp')
    # ax2.grid(True)

    # # Subplot 3: Required Base Position
    # ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    # ax3.plot3D(P_base[0, :], P_base[1, :], P_base[2, :], 'b', linewidth=2)
    # ax3.quiver(P_base[0, :], P_base[1, :], P_base[2, :], O_base[0, :], O_base[1, :], O_base[2, :], color='r', length=0.2)
    # ax3.quiver(P_base[0, :], P_base[1, :], P_base[2, :], O_t_base[0, :], O_t_base[1, :], O_t_base[2, :], color='g', length=0.2)
    # ax3.quiver(P_base[0, :], P_base[1, :], P_base[2, :], N_base[0, :], N_base[1, :], N_base[2, :], color='y', length=0.2)
    # ax3.set_title('Required Base\nposition and orientation')
    # ax3.set_xlabel('x_base')
    # ax3.set_ylabel('y_base')
    # ax3.set_zlabel('z_base')
    # ax3.grid(True)

    plt.tight_layout()
    plt.ion()
    plt.show(block=True)    
    print(f"Episode {ep + 1} finished with Total Reward: {episode_reward:.2f}")
    print(f"Saving test_envOne_ep{ep + 1}.mp4...")
    #imageio.mimsave(f"test_envOne_ep{ep + 1}.mp4", frames, fps=30)
print(positions)
print("Done! You can now watch test_env4.mp4")
# 5. Clean up
env.close()
print("Testing complete.")
#%%
# import matplotlib.pyplot as plt
# import numpy as np
# fig = plt.figure(figsize=(18, 6))
# positions = [np.array([ 240.18, -243.96,   66.58]), np.array([ 241.03, -244.4 ,   67.4 ]), np.array([ 251.54, -242.12,   74.57]), np.array([ 251.49, -242.08,   74.89]), np.array([ 245.88, -240.06,   67.53]), np.array([ 246.56, -241.35,   68.41]), np.array([ 249.17, -242.81,   72.98]), np.array([ 247.45, -243.74,   71.88]), np.array([ 246.3 , -245.03,   70.76]), np.array([ 246.93, -245.6 ,   70.4 ]), np.array([ 246.98, -245.96,   70.02]), np.array([ 246.99, -245.98,   70.01]), np.array([ 247.27, -246.77,   70.03]), np.array([ 247.17, -246.84,   70.08]), np.array([ 247.15, -246.87,   70.01]), np.array([ 247.16, -246.84,   70.04]), np.array([ 247.17, -246.84,   70.01]), np.array([ 247.18, -246.82,   70.03]), np.array([ 247.16, -246.82,   70.02]), np.array([ 247.13, -246.79,   70.  ]), np.array([ 247.13, -246.78,   69.99]), np.array([ 247.14, -246.77,   70.  ]), np.array([ 247.14, -246.76,   70.  ]), np.array([ 247.17, -246.77,   70.02]), np.array([ 247.16, -246.75,   70.01]), np.array([ 247.18, -246.76,   70.  ]), np.array([ 247.16, -246.76,   69.99]), np.array([ 247.13, -246.75,   69.98]), np.array([ 247.13, -246.74,   69.95]), np.array([ 247.15, -246.78,   69.96]), np.array([ 247.18, -246.78,   70.  ]), np.array([ 247.17, -246.79,   70.  ]), np.array([ 247.17, -246.81,   69.99]), np.array([ 247.17, -246.8 ,   69.99]), np.array([ 247.66, -247.45,   69.97]), np.array([ 247.59, -247.46,   70.03]), np.array([ 247.62, -247.48,   70.02]), np.array([ 247.63, -247.46,   70.04]), np.array([ 247.63, -247.47,   70.03]), np.array([ 247.64, -247.46,   70.04]), np.array([ 247.66, -247.47,   70.03]), np.array([ 247.66, -247.48,   70.03]), np.array([ 247.66, -247.46,   70.03]), np.array([ 247.64, -247.47,   70.02]), np.array([ 247.65, -247.45,   70.  ]), np.array([ 247.63, -247.46,   69.98]), np.array([ 247.63, -247.48,   70.  ]), np.array([ 247.65, -247.48,   70.  ]), np.array([ 247.66, -247.5 ,   69.99]), np.array([ 247.64, -247.49,   69.98]), np.array([ 248.29, -248.02,   70.03]), np.array([ 248.27, -248.  ,   70.05]), np.array([ 248.27, -248.  ,   70.05]), np.array([ 248.29, -248.  ,   70.05]), np.array([ 248.28, -248.  ,   70.04]), np.array([ 248.27, -248.  ,   70.03]), np.array([ 248.28, -247.98,   70.01]), np.array([ 248.28, -248.  ,   70.01]), np.array([ 248.27, -248.  ,   70.  ]), np.array([ 248.28, -248.  ,   70.  ]), np.array([ 248.25, -248.01,   69.99]), np.array([ 248.95, -248.37,   70.01]), np.array([ 248.92, -248.38,   70.02]), np.array([ 248.94, -248.36,   70.02]), np.array([ 248.93, -248.36,   70.01]), np.array([ 248.95, -248.38,   70.01]), np.array([ 248.94, -248.36,   70.01]), np.array([ 248.93, -248.38,   70.  ]), np.array([ 248.93, -248.38,   70.  ]), np.array([ 249.56, -248.64,   69.96]), np.array([ 249.58, -248.65,   69.97]), np.array([ 249.58, -248.62,   70.01]), np.array([ 249.58, -248.61,   70.01]), np.array([ 249.57, -248.62,   70.  ]), np.array([ 250.19, -248.79,   69.98]), np.array([ 250.19, -248.81,   69.99]), np.array([ 250.17, -248.78,   70.  ]), np.array([ 250.17, -248.79,   70.  ]), np.array([ 250.18, -248.78,   70.  ]), np.array([ 250.75, -248.9 ,   70.  ]), np.array([ 250.73, -248.89,   70.  ]), np.array([ 250.73, -248.89,   70.  ]), np.array([ 250.71, -248.88,   70.  ]), np.array([ 251.25, -248.94,   70.03]), np.array([ 251.24, -248.96,   70.  ]), np.array([ 251.22, -248.99,   69.98]), np.array([ 251.77, -248.98,   70.02]), np.array([ 251.78, -249.01,   70.  ]), np.array([ 251.74, -248.99,   70.01]), np.array([ 252.25, -248.98,   70.  ]), np.array([ 252.25, -248.99,   70.01]), np.array([ 252.25, -248.97,   70.  ]), np.array([ 252.76, -248.95,   70.01]), np.array([ 252.76, -248.95,   70.  ]), np.array([ 252.75, -248.96,   70.  ]), np.array([ 253.28, -248.88,   70.  ]), np.array([ 253.27, -248.89,   70.  ]), np.array([ 253.27, -248.88,   70.  ]), np.array([ 253.25, -248.88,   70.  ]), np.array([ 253.81, -248.81,   70.  ]), np.array([ 253.81, -248.77,   70.  ]), np.array([ 253.83, -248.77,   70.  ]), np.array([ 253.8 , -248.77,   70.01]), np.array([ 254.41, -248.67,   69.99]), np.array([ 254.41, -248.61,   70.  ]), np.array([ 254.42, -248.62,   70.01]), np.array([ 254.4 , -248.62,   69.99]), np.array([ 255.04, -248.4 ,   69.96]), np.array([ 255.03, -248.38,   69.97]), np.array([ 255.05, -248.38,   69.97]), np.array([ 255.06, -248.37,   70.  ]), np.array([ 255.05, -248.38,   70.  ]), np.array([ 255.1 , -248.4 ,   69.99]), np.array([ 255.69, -248.02,   70.02]), np.array([ 255.66, -248.  ,   70.  ]), np.array([ 255.68, -248.  ,   69.99]), np.array([ 255.68, -248.01,   69.99]), np.array([ 255.71, -247.99,   69.99]), np.array([ 255.71, -248.  ,   70.01]), np.array([ 255.7 , -247.99,   69.99]), np.array([ 255.73, -248.01,   70.02]), np.array([ 256.32, -247.5 ,   70.03]), np.array([ 256.3 , -247.5 ,   70.02]), np.array([ 256.3 , -247.49,   70.02]), np.array([ 256.3 , -247.49,   70.01]), np.array([ 256.31, -247.48,   69.99]), np.array([ 256.31, -247.48,   70.  ]), np.array([ 256.32, -247.46,   70.  ]), np.array([ 256.33, -247.48,   70.  ]), np.array([ 256.34, -247.46,   70.01]), np.array([ 256.34, -247.48,   70.  ]), np.array([ 256.33, -247.47,   69.99]), np.array([ 256.53, -247.39,   70.06]), np.array([ 256.82, -246.83,   70.03]), np.array([ 256.82, -246.82,   70.02]), np.array([ 256.82, -246.81,   70.03]), np.array([ 256.82, -246.82,   70.02]), np.array([ 256.81, -246.82,   70.02]), np.array([ 256.82, -246.82,   70.02]), np.array([ 256.81, -246.82,   70.01]), np.array([ 256.82, -246.8 ,   70.02]), np.array([ 256.83, -246.8 ,   70.  ]), np.array([ 256.83, -246.79,   70.01]), np.array([ 256.83, -246.78,   70.01]), np.array([ 256.83, -246.78,   70.01]), np.array([ 256.84, -246.78,   70.01]), np.array([ 256.85, -246.79,   70.01]), np.array([ 256.85, -246.81,   70.01]), np.array([ 256.8 , -246.77,   69.99]), np.array([ 257.07, -246.69,   70.05]), np.array([ 256.82, -246.79,   70.  ]), np.array([ 256.83, -246.79,   69.92]), np.array([ 256.79, -246.77,   70.  ]), np.array([ 256.84, -246.79,   69.94]), np.array([ 257.04, -245.96,   70.03]), np.array([ 257.04, -245.99,   70.03]), np.array([ 257.03, -246.01,   70.04]), np.array([ 257.02, -246.  ,   70.02]), np.array([ 257.03, -246.  ,   70.03]), np.array([ 257.03, -246.  ,   70.03]), np.array([ 257.02, -246.  ,   70.02]), np.array([ 257.02, -246.01,   70.02]), np.array([ 257.02, -246.01,   70.02]), np.array([ 257.02, -246.01,   70.01]), np.array([ 257.02, -246.01,   70.02]), np.array([ 257.03, -246.  ,   70.01]), np.array([ 257.04, -246.  ,   70.01]), np.array([ 257.04, -245.99,   70.01]), np.array([ 257.04, -245.99,   70.02]), np.array([ 257.03, -245.98,   70.02]), np.array([ 257.04, -245.98,   70.02]), np.array([ 257.05, -245.99,   70.02]), np.array([ 257.05, -245.99,   70.02]), np.array([ 257.05, -246.03,   70.02]), np.array([ 257.06, -246.05,   70.02]), np.array([ 257.05, -246.06,   70.03]), np.array([ 257.05, -246.07,   70.04]), np.array([ 257.04, -246.05,   70.04]), np.array([ 257.05, -246.05,   70.03]), np.array([ 257.05, -246.05,   70.03]), np.array([ 257.05, -246.06,   70.04]), np.array([ 257.04, -246.07,   70.03]), np.array([ 257.05, -246.06,   70.04]), np.array([ 257.04, -246.07,   70.04]), np.array([ 257.03, -246.07,   70.03]), np.array([ 257.04, -246.08,   70.03]), np.array([ 257.03, -246.08,   70.04]), np.array([ 257.03, -246.09,   70.03]), np.array([ 257.03, -246.09,   70.04]), np.array([ 257.02, -246.1 ,   70.04]), np.array([ 257.03, -246.1 ,   70.03]), np.array([ 257.02, -246.11,   70.04]), np.array([ 257.02, -246.12,   70.04]), np.array([ 257.02, -246.12,   70.05]), np.array([ 257.02, -246.12,   70.04]), np.array([ 257.01, -246.12,   70.04]), np.array([ 257.02, -246.12,   70.05]), np.array([ 257.03, -246.12,   70.04]), np.array([ 257.02, -246.13,   70.05]), np.array([ 257.02, -246.12,   70.05]), np.array([ 257.03, -246.13,   70.05]), np.array([ 257.02, -246.13,   70.05]), np.array([ 257.03, -246.13,   70.05]), np.array([ 257.02, -246.13,   70.05]), np.array([ 257.03, -246.13,   70.05]), np.array([ 257.03, -246.13,   70.05]), np.array([ 257.03, -246.13,   70.06]), np.array([ 257.03, -246.13,   70.05]), np.array([ 257.03, -246.13,   70.06]), np.array([ 257.03, -246.13,   70.05]), np.array([ 257.03, -246.13,   70.06]), np.array([ 257.04, -246.13,   70.05]), np.array([ 257.04, -246.13,   70.06]), np.array([ 257.04, -246.14,   70.06]), np.array([ 257.03, -246.13,   70.06]), np.array([ 257.04, -246.13,   70.06]), np.array([ 257.04, -246.13,   70.06]), np.array([ 257.05, -246.13,   70.06]), np.array([ 257.04, -246.13,   70.06]), np.array([ 257.04, -246.13,   70.07]), np.array([ 257.05, -246.13,   70.06]), np.array([ 257.04, -246.12,   70.06]), np.array([ 257.05, -246.13,   70.07]), np.array([ 257.05, -246.13,   70.06]), np.array([ 257.05, -246.12,   70.06]), np.array([ 257.04, -246.13,   70.07]), np.array([ 257.06, -246.13,   70.06]), np.array([ 257.05, -246.12,   70.07]), np.array([ 257.05, -246.13,   70.07]), np.array([ 257.05, -246.12,   70.06]), np.array([ 257.05, -246.12,   70.07]), np.array([ 257.05, -246.13,   70.07]), np.array([ 257.06, -246.12,   70.07]), np.array([ 257.05, -246.13,   70.07]), np.array([ 257.06, -246.12,   70.07]), np.array([ 257.05, -246.12,   70.07]), np.array([ 257.06, -246.12,   70.07]), np.array([ 257.06, -246.13,   70.07]), np.array([ 257.05, -246.12,   70.07]), np.array([ 257.06, -246.12,   70.07]), np.array([ 257.06, -246.12,   70.08]), np.array([ 257.05, -246.12,   70.07]), np.array([ 257.06, -246.12,   70.07]), np.array([ 257.06, -246.12,   70.08]), np.array([ 257.06, -246.12,   70.07]), np.array([ 257.06, -246.12,   70.08]), np.array([ 257.06, -246.12,   70.07]), np.array([ 257.06, -246.12,   70.08]), np.array([ 257.06, -246.12,   70.07]), np.array([ 257.07, -246.12,   70.08]), np.array([ 257.06, -246.12,   70.08]), np.array([ 257.06, -246.11,   70.08]), np.array([ 257.07, -246.12,   70.07]), np.array([ 257.05, -246.12,   70.08]), np.array([ 257.07, -246.12,   70.08]), np.array([ 257.06, -246.11,   70.08]), np.array([ 257.07, -246.12,   70.08]), np.array([ 257.06, -246.12,   70.08]), np.array([ 247.16, -250.63,   72.63]), np.array([ 246.76, -249.12,   71.56])]
# positions = np.array([a.tolist() for a in positions])
# # Subplot 1: Target Needle Tip
# ax1 = fig.add_subplot(1, 3, 1, projection='3d')
# ax1.scatter3D(positions[:, 0], positions[:, 1], positions[:, 2], 'b', linewidth=2)
# #ax1.quiver(P_t[0, :], P_t[1, :], P_t[2, :], O_t[0, :], O_t[1, :], O_t[2, :], color='r', length=0.2)
# ax1.set_title('Target Needle Tip\n(global pos + global orient)')
# ax1.set_xlabel('x_t')
# ax1.set_ylabel('y_t')
# ax1.set_zlabel('z_t')
# #ax1.view_init(elev=90, azim=0) # Emulates view(90,0)
# ax1.grid(True)

# # Subplot 2: Target TCP
# # ax2 = fig.add_subplot(1, 3, 2, projection='3d')
# # ax2.plot3D(P_tcp[0, :], P_tcp[1, :], P_tcp[2, :], 'b', linewidth=2)
# # ax2.quiver(P_tcp[0, :], P_tcp[1, :], P_tcp[2, :], O_tcp[0, :], O_tcp[1, :], O_tcp[2, :], color='r', length=0.2)
# # ax2.set_title('Target tcp\n(global position)')
# # ax2.set_xlabel('x_tcp')
# # ax2.set_ylabel('y_tcp')
# # ax2.set_zlabel('z_tcp')
# # ax2.grid(True)

# # # Subplot 3: Required Base Position
# # ax3 = fig.add_subplot(1, 3, 3, projection='3d')
# # ax3.plot3D(P_base[0, :], P_base[1, :], P_base[2, :], 'b', linewidth=2)
# # ax3.quiver(P_base[0, :], P_base[1, :], P_base[2, :], O_base[0, :], O_base[1, :], O_base[2, :], color='r', length=0.2)
# # ax3.quiver(P_base[0, :], P_base[1, :], P_base[2, :], O_t_base[0, :], O_t_base[1, :], O_t_base[2, :], color='g', length=0.2)
# # ax3.quiver(P_base[0, :], P_base[1, :], P_base[2, :], N_base[0, :], N_base[1, :], N_base[2, :], color='y', length=0.2)
# # ax3.set_title('Required Base\nposition and orientation')
# # ax3.set_xlabel('x_base')
# # ax3.set_ylabel('y_base')
# # ax3.set_zlabel('z_base')
# # ax3.grid(True)

# plt.tight_layout()
# plt.ion()
# plt.show()
# %%
