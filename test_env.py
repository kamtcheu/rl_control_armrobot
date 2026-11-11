import time

import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.utils import EzPickle
import sys
import traceback
import signal

def keyboard_interrupt_handler(sig, frame, rob1):
    print('KeyboardInterrupt detected. Exiting.')
    try:
        rob1.close()
    except NameError:
        pass
    # try:
    #     rob2.close()
    # except NameError:
    #     pass
    traceback.print_stack()
    sys.exit(0)

signal.signal(signal.SIGINT, lambda sig, frame: keyboard_interrupt_handler(sig, frame, rob1))

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
import mujoco
from stable_baselines3 import SAC 
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from theEnv31 import ArmRobot, clean_null_noise, orientation_error_matrices, orientation_error_quaternions
from trajectory1 import T_tcp2t, R_tcp2t, T_t2tcp, T_global2base
from urx import *
import vectorfunctions as vf
# os.environ["DISPLAY"] = "0"
# # os.environ["MUJOCO_GL"] = "egl"
def draw_box(ax, x0, y0, z0, L=5, color="blue"):
    h = L / 2

    # Eight vertices of the cube
    vertices = np.array([
        [x0 + dx, y0 + dy, z0 + dz]
        for dx in (-h, h)
        for dy in (-h, h)
        for dz in (-h, h)
    ])

    # Connect vertices differing in exactly one coordinate
    for i in range(8):
        for j in range(i + 1, 8):
            if np.count_nonzero(vertices[i] != vertices[j]) == 1:
                ax.plot(
                    *zip(vertices[i], vertices[j]),
                    color=color,
                    linewidth=1.5
                )

def interpolate_joint_trajectory(q_start, q_goal, T, dt):
    q_start = np.asarray(q_start)
    q_goal = np.asarray(q_goal)

    times = np.arange(0, T + dt, dt)

    trajectory = []

    for t in times:
        tau = np.clip(t / T, 0.0, 1.0)

        # Quintic time scaling
        s = 10*tau**3 - 15*tau**4 + 6*tau**5

        q = q_start + s * (q_goal - q_start)

        trajectory.append(q)

    return np.array(trajectory)
model_path = "sac_eval_checkpoints/nolook_envTv36_sac_eval_4000000_steps" 
robot=False
# 2. Initialize the environment with human rendering
# render_mode="human" automatically opens a window and renders each step, np.array([[-335 - 0.07], [-10 ], [+295 + 0.014]])
env = ArmRobot(render_mode="human",ellipse_center_g=np.array([[-335], [-10 ], [+295]]), a=5,b=3,start_target_idx=1,ori_cost=1.5,threshold_pos=50e-6, threshold_ori=1.5, max_joint_delta=0.4, qpos_noise=False,episode_len=250, num_points=20, work_box_size=0.1)#null_threshold=1e-8, work_box_size=0.1, threshold_pos=50e-6, threshold_ori=1.5, start_target_idx=0, max_joint_delta = 3.0, render_mode=None, site_name="needle_holder_site", episode_len=250, num_points=20)
if robot:
    rob1 = Robot("192.168.56.101")
    #rob1.set_gravity((0.0, + 9.81, 0.0))
    rob1.movej((-0.37626284, -1.71089287, -1.15919388, -1.85352342,  1.55074358, -0.37225355), vel=1.0)

#env._max_joint_delta[-1]=15
#env = gym.make(env_name, render_mode="human")
# gym.register(id="my_master_envs/ArmRobot-v4", entry_point="trajectory_env4:ArmRobot")
# env = gym.make("my_master_envs/ArmRobot-v4", null_threshold=1e-8, work_box_size=0.1, threshold_pos=50e-6, threshold_ori=1.50, start_target_idx=0, max_joint_delta = 0.5, render_mode=None, site_name="needle_holder_site", episode_len=250, num_points=20)
#env = gym.wrappers.RecordVideo(env=env, video_folder="./saved_models")
# 3. Load the trained model
#env.reset()
# Note: Ensure you import the exact algorithm class you used for training
print(f"Loading model from {model_path}...")
model = SAC.load(os.path.abspath(model_path), env=env)
best_action = False
# 4. Evaluation Loop
num_episodes = 10
positions = []
orientations = []
orientations_int = []
ori_errors = []
cross_error_vecs = []
frames = []

for ep in range(num_episodes):
    # Reset the environment at the start of each episode
    obs, info = env.reset()
    p = info["current_pos"]
    if robot:
        rob1 = Robot("192.168.56.101")
        rob1.movej((-0.37626284, -1.71089287, -1.15919388, -1.85352342,  1.55074358, -0.37225355), vel=1.0, acc=0.5)

    done = False
    truncated = False
    episode_reward = 0.0
    position = []
    orientation = []
    orientation_int = []
    ori_error = []
    cross_error_vec = []
    frames = []
    actions = []
    print(f"Starting Episode {ep + 1}...")
    position.append(info["current_pos"])
    orientation.append(info["current_ori"])
    orientation_int.append(info["R_int"])
    cross_error_vec.append(float(info["cross_error_vec"]))
    ori_error.append(float(info["cos_theta"]))

    # Run the episode until the agent dies (done) or hits the time limit (truncated)
    while not (done or truncated):#):# 
        # Predict the action to take. 
        # deterministic=True is usually best for testing/evaluating trained agents.
        action, _states = model.predict(obs, deterministic=True)
        
        #actions.append(action)
        print(action)
        # Take a step in the environment
        obs, reward, done, truncated, info = env.step(action)
        print("obs env: ", obs)
        # if robot:
        #     ac = np.deg2rad(np.clip(action, -1.0, 1.0) * env._max_joint_delta )
        #     abs_ac = rob1.getj() + ac
        #     clipped_action = np.clip(
        #                 abs_ac, 
        #                 env.actions_low, 
        #                 env.actions_high
        #     )
        #     clipped_action = np.round(clipped_action, 4)
        #     rob1.movej(env.data.qpos.copy(), vel=0.1, acc=0.05)
        #     rob1_ori, rob1_pos = vf.extract_data_from_robot_pose(rob1.get_pose())
        #     rob1_pos = np.array([a[0] for a in rob1_pos])
        #     rob1_ori = np.array(rob1_ori) #np.array([rob1_ori[:,0], rob1_ori[:,1], rob1_ori[:,2]])
        #     #rob1_pos = np.array([rob1.x, rob1.y, rob1.z])
        #     print("rob1_pos: ", rob1_pos)

        #     segment = env.current_segment
        #     to_robot = rob1_pos - env.prev_target_pos
    
        #     segment_sq = np.dot(segment, segment)
        #     assert segment_sq > 1e-8 , "current segment length is null. not a valid value"
        #     #if segment_sq > 1e-8:
        #     s = np.dot(to_robot, segment) / segment_sq
        #     print("s: ", s)
        #     s_clip = np.clip(s, 0.0, 1.0)
        #     p_proj = env.prev_target_pos + s_clip * env.current_segment
        #     deviation_error_vec = rob1_pos - p_proj
        #     seg = np.array(segment)/np.linalg.norm(segment)

        #     err_vec_pos = np.round(np.array(env.current_target_pos) - np.array(rob1_pos), 2)


        #     #rob1_ori = np.array(rob1.get_orientation())
        #     print("rob1_ori: ", rob1_ori)
        #     R_des = env.interpolate_rotation(env.prev_target_ori, env.current_target_ori, s)
        #     R_err = R_des.T @ rob1_ori

        #     mat = np.dot(rob1_ori.T, env.current_target_ori)
        #     err_mat = np.round(clean_null_noise(mat, env.null_threshold), 3)
        #     # Access the raw dictionary of the latest packet data
        #     joint_data = rob1.secmon.get_joint_data()

        #     # Extract actual joint velocities (q')
        #     q_prime = [
        #         joint_data["qd_actual0"],
        #         joint_data["qd_actual1"],
        #         joint_data["qd_actual2"],
        #         joint_data["qd_actual3"],
        #         joint_data["qd_actual4"],
        #         joint_data["qd_actual5"]
        #     ]
        #     obs = np.concatenate([
        #                         # joint positions, velocities and control signals
        #                         np.array(rob1.getj()),
        #                         np.array(q_prime),
        #                         # cartesian postions error
        #                         err_vec_pos,
        #                         np.array([s]),
        #                         np.array(deviation_error_vec),
        #                         seg,
        #                         # orientation of the end effector
        #                         np.array(err_mat[:,:2].T.flatten()),
        #                         np.array(R_err[:,:2].T.flatten()), # 6D representation of 3 x 3 orientation (2 first column). 
        #                                                                         #the last column must be the result of the cross product of 1st column(tangent) with second(binormal)
        #                         #TODO: Add a look ahead mechanism
        #                         # self.get_lookahead_target().flatten(),
                                
        #                         ],
        #                         axis=0)
        #     print("obs robot: ", obs)
        episode_reward += reward
        position.append(info["current_pos"])
        orientation.append(info["current_ori"])
        orientation_int.append(info["R_int"])
        ori_error.append(float(info["cos_theta"]))
        cross_error_vec.append(float(info["cross_error_vec"]))
        actions.append(env.data.qpos.copy())
        #actions.append(action)
    print("len actions: ", len(actions))
    if robot:
        rob1.movejs(actions, vel=1.0, acc=0.5)
        # for i, act in enumerate(actions):
        #     #crtl = rob1.getj()
        #     ac = np.deg2rad(np.clip(act, -1.0, 1.0) * env._max_joint_delta )
        #     abs_ac = rob1.getj() + ac
        #     clipped_action = np.clip(
        #                 abs_ac, 
        #                 env.actions_low, 
        #                 env.actions_high
        #     )
        #     clipped_action = np.round(clipped_action, 4)
        #     rob1.movej(clipped_action)
        #     print("Current step positions: ", i)
        rob1.close()
    if best_action:
        actions = env.best_actions_list.copy()
        best_steps = env.best_step_number_l.copy()
        print("Action's list: ", env.best_actions_list)
        print("stepnumber's list: ", env.best_step_number_l)
        obs, info = env.reset()
        timest=0
        total_trajectory = []
        a_ = env.data.qpos[:6].copy()
        total_trajectory.append(a_)
        for num, a in enumerate(actions):
            #obs, reward, done, truncated, info = env.step(a)
            #env.do_simulation(a, env.frame_skip)
            #if timest == 0:
            # env.data.qpos[:6] = a
            # mujoco.mj_forward(env.model, env.data)

            # if env.render_mode == "human":
            #     env.render()
            #     time.sleep(0.3)
            #else:
            dt_ = 0.1
            if num/10 <= 0.5 or num/10 >= 1.5:
                dt_ = 0.2
            else:
                dt_ = 0.4
            tr = interpolate_joint_trajectory(a_, a, T=0.5, dt=dt_)
            for q in tr[1:]:
                total_trajectory.append(q)
            a_ = a
        for q in total_trajectory:
            env.data.qpos[:6] = q
            mujoco.mj_forward(env.model, env.data)
            if env.render_mode == "human":
                env.render()
                #time.sleep(0.01)
            #timest += 1
        position = [position[i] for i in best_steps]
        orientation = [orientation[i] for i in best_steps]
        orientation_int = [orientation_int[i] for i in best_steps]
        ori_error = [ori_error[i] for i in best_steps]
        cross_error_vec = [cross_error_vec[i] for i in best_steps]

        
        #frames.append(env.render())
    print(len(list(range(len(cross_error_vec)))))
    print(len(position))
    assert len(position)>0 and len(position) == len(orientation) == len(cross_error_vec), "Lengths of positions, orientations, and cross_error_vecs do not match."
    import matplotlib.pyplot as plt
    import numpy as np 
    fig = plt.figure(figsize=(18, 6))
    filt=1
    #positions = #[np.array([ 240.18, -243.96,   66.58]), np.array([ 241.03, -244.4 ,   67.4 ]), np.array([ 251.54, -242.12,   74.57]), np.array([ 251.49, -242.08,   74.89]), np.array([ 245.88, -240.06,   67.53]), np.array([ 246.56, -241.35,   68.41]), np.array([ 249.17, -242.81,   72.98]), np.array([ 247.45, -243.74,   71.88]), np.array([ 246.3 , -245.03,   70.76]), np.array([ 246.93, -245.6 ,   70.4 ]), np.array([ 246.98, -245.96,   70.02]), np.array([ 246.99, -245.98,   70.01]), np.array([ 247.27, -246.77,   70.03]), np.array([ 247.17, -246.84,   70.08]), np.array([ 247.15, -246.87,   70.01]), np.array([ 247.16, -246.84,   70.04]), np.array([ 247.17, -246.84,   70.01]), np.array([ 247.18, -246.82,   70.03]), np.array([ 247.16, -246.82,   70.02]), np.array([ 247.13, -246.79,   70.  ]), np.array([ 247.13, -246.78,   69.99]), np.array([ 247.14, -246.77,   70.  ]), np.array([ 247.14, -246.76,   70.  ]), np.array([ 247.17, -246.77,   70.02]), np.array([ 247.16, -246.75,   70.01]), np.array([ 247.18, -246.76,   70.  ]), np.array([ 247.16, -246.76,   69.99]), np.array([ 247.13, -246.75,   69.98]), np.array([ 247.13, -246.74,   69.95]), np.array([ 247.15, -246.78,   69.96]), np.array([ 247.18, -246.78,   70.  ]), np.array([ 247.17, -246.79,   70.  ]), np.array([ 247.17, -246.81,   69.99]), np.array([ 247.17, -246.8 ,   69.99]), np.array([ 247.66, -247.45,   69.97]), np.array([ 247.59, -247.46,   70.03]), np.array([ 247.62, -247.48,   70.02]), np.array([ 247.63, -247.46,   70.04]), np.array([ 247.63, -247.47,   70.03]), np.array([ 247.64, -247.46,   70.04]), np.array([ 247.66, -247.47,   70.03]), np.array([ 247.66, -247.48,   70.03]), np.array([ 247.66, -247.46,   70.03]), np.array([ 247.64, -247.47,   70.02]), np.array([ 247.65, -247.45,   70.  ]), np.array([ 247.63, -247.46,   69.98]), np.array([ 247.63, -247.48,   70.  ]), np.array([ 247.65, -247.48,   70.  ]), np.array([ 247.66, -247.5 ,   69.99]), np.array([ 247.64, -247.49,   69.98]), np.array([ 248.29, -248.02,   70.03]), np.array([ 248.27, -248.  ,   70.05]), np.array([ 248.27, -248.  ,   70.05]), np.array([ 248.29, -248.  ,   70.05]), np.array([ 248.28, -248.  ,   70.04]), np.array([ 248.27, -248.  ,   70.03]), np.array([ 248.28, -247.98,   70.01]), np.array([ 248.28, -248.  ,   70.01]), np.array([ 248.27, -248.  ,   70.  ]), np.array([ 248.28, -248.  ,   70.  ]), np.array([ 248.25, -248.01,   69.99]), np.array([ 248.95, -248.37,   70.01]), np.array([ 248.92, -248.38,   70.02]), np.array([ 248.94, -248.36,   70.02]), np.array([ 248.93, -248.36,   70.01]), np.array([ 248.95, -248.38,   70.01]), np.array([ 248.94, -248.36,   70.01]), np.array([ 248.93, -248.38,   70.  ]), np.array([ 248.93, -248.38,   70.  ]), np.array([ 249.56, -248.64,   69.96]), np.array([ 249.58, -248.65,   69.97]), np.array([ 249.58, -248.62,   70.01]), np.array([ 249.58, -248.61,   70.01]), np.array([ 249.57, -248.62,   70.  ]), np.array([ 250.19, -248.79,   69.98]), np.array([ 250.19, -248.81,   69.99]), np.array([ 250.17, -248.78,   70.  ]), np.array([ 250.17, -248.79,   70.  ]), np.array([ 250.18, -248.78,   70.  ]), np.array([ 250.75, -248.9 ,   70.  ]), np.array([ 250.73, -248.89,   70.  ]), np.array([ 250.73, -248.89,   70.  ]), np.array([ 250.71, -248.88,   70.  ]), np.array([ 251.25, -248.94,   70.03]), np.array([ 251.24, -248.96,   70.  ]), np.array([ 251.22, -248.99,   69.98]), np.array([ 251.77, -248.98,   70.02]), np.array([ 251.78, -249.01,   70.  ]), np.array([ 251.74, -248.99,   70.01]), np.array([ 252.25, -248.98,   70.  ]), np.array([ 252.25, -248.99,   70.01]), np.array([ 252.25, -248.97,   70.  ]), np.array([ 252.76, -248.95,   70.01]), np.array([ 252.76, -248.95,   70.  ]), np.array([ 252.75, -248.96,   70.  ]), np.array([ 253.28, -248.88,   70.  ]), np.array([ 253.27, -248.89,   70.  ]), np.array([ 253.27, -248.88,   70.  ]), np.array([ 253.25, -248.88,   70.  ]), np.array([ 253.81, -248.81,   70.  ]), np.array([ 253.81, -248.77,   70.  ]), np.array([ 253.83, -248.77,   70.  ]), np.array([ 253.8 , -248.77,   70.01]), np.array([ 254.41, -248.67,   69.99]), np.array([ 254.41, -248.61,   70.  ]), np.array([ 254.42, -248.62,   70.01]), np.array([ 254.4 , -248.62,   69.99]), np.array([ 255.04, -248.4 ,   69.96]), np.array([ 255.03, -248.38,   69.97]), np.array([ 255.05, -248.38,   69.97]), np.array([ 255.06, -248.37,   70.  ]), np.array([ 255.05, -248.38,   70.  ]), np.array([ 255.1 , -248.4 ,   69.99]), np.array([ 255.69, -248.02,   70.02]), np.array([ 255.66, -248.  ,   70.  ]), np.array([ 255.68, -248.  ,   69.99]), np.array([ 255.68, -248.01,   69.99]), np.array([ 255.71, -247.99,   69.99]), np.array([ 255.71, -248.  ,   70.01]), np.array([ 255.7 , -247.99,   69.99]), np.array([ 255.73, -248.01,   70.02]), np.array([ 256.32, -247.5 ,   70.03]), np.array([ 256.3 , -247.5 ,   70.02]), np.array([ 256.3 , -247.49,   70.02]), np.array([ 256.3 , -247.49,   70.01]), np.array([ 256.31, -247.48,   69.99]), np.array([ 256.31, -247.48,   70.  ]), np.array([ 256.32, -247.46,   70.  ]), np.array([ 256.33, -247.48,   70.  ]), np.array([ 256.34, -247.46,   70.01]), np.array([ 256.34, -247.48,   70.  ]), np.array([ 256.33, -247.47,   69.99]), np.array([ 256.53, -247.39,   70.06]), np.array([ 256.82, -246.83,   70.03]), np.array([ 256.82, -246.82,   70.02]), np.array([ 256.82, -246.81,   70.03]), np.array([ 256.82, -246.82,   70.02]), np.array([ 256.81, -246.82,   70.02]), np.array([ 256.82, -246.82,   70.02]), np.array([ 256.81, -246.82,   70.01]), np.array([ 256.82, -246.8 ,   70.02]), np.array([ 256.83, -246.8 ,   70.  ]), np.array([ 256.83, -246.79,   70.01]), np.array([ 256.83, -246.78,   70.01]), np.array([ 256.83, -246.78,   70.01]), np.array([ 256.84, -246.78,   70.01]), np.array([ 256.85, -246.79,   70.01]), np.array([ 256.85, -246.81,   70.01]), np.array([ 256.8 , -246.77,   69.99]), np.array([ 257.07, -246.69,   70.05]), np.array([ 256.82, -246.79,   70.  ]), np.array([ 256.83, -246.79,   69.92]), np.array([ 256.79, -246.77,   70.  ]), np.array([ 256.84, -246.79,   69.94]), np.array([ 257.04, -245.96,   70.03]), np.array([ 257.04, -245.99,   70.03]), np.array([ 257.03, -246.01,   70.04]), np.array([ 257.02, -246.  ,   70.02]), np.array([ 257.03, -246.  ,   70.03]), np.array([ 257.03, -246.  ,   70.03]), np.array([ 257.02, -246.  ,   70.02]), np.array([ 257.02, -246.01,   70.02]), np.array([ 257.02, -246.01,   70.02]), np.array([ 257.02, -246.01,   70.01]), np.array([ 257.02, -246.01,   70.02]), np.array([ 257.03, -246.  ,   70.01]), np.array([ 257.04, -246.  ,   70.01]), np.array([ 257.04, -245.99,   70.01]), np.array([ 257.04, -245.99,   70.02]), np.array([ 257.03, -245.98,   70.02]), np.array([ 257.04, -245.98,   70.02]), np.array([ 257.05, -245.99,   70.02]), np.array([ 257.05, -245.99,   70.02]), np.array([ 257.05, -246.03,   70.02]), np.array([ 257.06, -246.05,   70.02]), np.array([ 257.05, -246.06,   70.03]), np.array([ 257.05, -246.07,   70.04]), np.array([ 257.04, -246.05,   70.04]), np.array([ 257.05, -246.05,   70.03]), np.array([ 257.05, -246.05,   70.03]), np.array([ 257.05, -246.06,   70.04]), np.array([ 257.04, -246.07,   70.03]), np.array([ 257.05, -246.06,   70.04]), np.array([ 257.04, -246.07,   70.04]), np.array([ 257.03, -246.07,   70.03]), np.array([ 257.04, -246.08,   70.03]), np.array([ 257.03, -246.08,   70.04]), np.array([ 257.03, -246.09,   70.03]), np.array([ 257.03, -246.09,   70.04]), np.array([ 257.02, -246.1 ,   70.04]), np.array([ 257.03, -246.1 ,   70.03]), np.array([ 257.02, -246.11,   70.04]), np.array([ 257.02, -246.12,   70.04]), np.array([ 257.02, -246.12,   70.05]), np.array([ 257.02, -246.12,   70.04]), np.array([ 257.01, -246.12,   70.04]), np.array([ 257.02, -246.12,   70.05]), np.array([ 257.03, -246.12,   70.04]), np.array([ 257.02, -246.13,   70.05]), np.array([ 257.02, -246.12,   70.05]), np.array([ 257.03, -246.13,   70.05]), np.array([ 257.02, -246.13,   70.05]), np.array([ 257.03, -246.13,   70.05]), np.array([ 257.02, -246.13,   70.05]), np.array([ 257.03, -246.13,   70.05]), np.array([ 257.03, -246.13,   70.05]), np.array([ 257.03, -246.13,   70.06]), np.array([ 257.03, -246.13,   70.05]), np.array([ 257.03, -246.13,   70.06]), np.array([ 257.03, -246.13,   70.05]), np.array([ 257.03, -246.13,   70.06]), np.array([ 257.04, -246.13,   70.05]), np.array([ 257.04, -246.13,   70.06]), np.array([ 257.04, -246.14,   70.06]), np.array([ 257.03, -246.13,   70.06]), np.array([ 257.04, -246.13,   70.06]), np.array([ 257.04, -246.13,   70.06]), np.array([ 257.05, -246.13,   70.06]), np.array([ 257.04, -246.13,   70.06]), np.array([ 257.04, -246.13,   70.07]), np.array([ 257.05, -246.13,   70.06]), np.array([ 257.04, -246.12,   70.06]), np.array([ 257.05, -246.13,   70.07]), np.array([ 257.05, -246.13,   70.06]), np.array([ 257.05, -246.12,   70.06]), np.array([ 257.04, -246.13,   70.07]), np.array([ 257.06, -246.13,   70.06]), np.array([ 257.05, -246.12,   70.07]), np.array([ 257.05, -246.13,   70.07]), np.array([ 257.05, -246.12,   70.06]), np.array([ 257.05, -246.12,   70.07]), np.array([ 257.05, -246.13,   70.07]), np.array([ 257.06, -246.12,   70.07]), np.array([ 257.05, -246.13,   70.07]), np.array([ 257.06, -246.12,   70.07]), np.array([ 257.05, -246.12,   70.07]), np.array([ 257.06, -246.12,   70.07]), np.array([ 257.06, -246.13,   70.07]), np.array([ 257.05, -246.12,   70.07]), np.array([ 257.06, -246.12,   70.07]), np.array([ 257.06, -246.12,   70.08]), np.array([ 257.05, -246.12,   70.07]), np.array([ 257.06, -246.12,   70.07]), np.array([ 257.06, -246.12,   70.08]), np.array([ 257.06, -246.12,   70.07]), np.array([ 257.06, -246.12,   70.08]), np.array([ 257.06, -246.12,   70.07]), np.array([ 257.06, -246.12,   70.08]), np.array([ 257.06, -246.12,   70.07]), np.array([ 257.07, -246.12,   70.08]), np.array([ 257.06, -246.12,   70.08]), np.array([ 257.06, -246.11,   70.08]), np.array([ 257.07, -246.12,   70.07]), np.array([ 257.05, -246.12,   70.08]), np.array([ 257.07, -246.12,   70.08]), np.array([ 257.06, -246.11,   70.08]), np.array([ 257.07, -246.12,   70.08]), np.array([ 257.06, -246.12,   70.08]), np.array([ 247.16, -250.63,   72.63]), np.array([ 246.76, -249.12,   71.56])]
    position = np.array([a.tolist() for i,a in enumerate(position) if i%filt==0]) # Downsample for plotting
    O_tcp = np.array([ori[:, 0] for i,ori in enumerate(orientation) if i%filt==0]).T
    O_tcp_int = np.array([ori[:, 0] for i,ori in enumerate(orientation_int) if i%filt==0]).T
    theta_list = np.array([np.rad2deg(np.arccos(a)) for i,a in enumerate(ori_error)])
    #theta_l = np.array([np.rad2deg(np.arccos(a)) for i,a in enumerate(ori_error) if i%filt==0])
    # Subplot 1: Target Needle Tip
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.plot3D(position[:, 0], position[:, 1], position[:, 2], 'b', linewidth=2)
    ax1.scatter3D(position[:, 0], position[:, 1], position[:, 2], 'b', linewidth=2, label="TCP Path")
    for i in range(len(position)):
        ax1.text(position[i, 0], position[i, 1], position[i, 2], str(i), zdir='z', fontsize=8)
    #ax1.text3D(positions[:, 0], positions[:, 1], positions[:, 2], [str(i) for i in range(len(positions))], zdir='z', fontsize=8)
    ax1.plot3D(env.target_path[0, :], env.target_path[1,: ], env.target_path[ 2,:], 'g', linewidth=2)
    ax1.scatter3D(env.target_path[0, :], env.target_path[1,: ], env.target_path[ 2,:], 'g', linewidth=2, label="Target TCP Path")
    ax1.quiver(position[:, 0], position[:, 1], position[:, 2], O_tcp[0, :], O_tcp[1, :], O_tcp[2, :], color='r', length=0.3, label="TCP Orientation")
    ax1.quiver(position[:, 0], position[:, 1], position[:, 2], O_tcp_int[0, :], O_tcp_int[1, :], O_tcp_int[2, :], color='black', length=0.3, label="TCP Orientation Target")
    #ax1.quiver(env.target_path[0, :], env.target_path[1,: ], env.target_path[ 2,:],env.target_tangent[0, :], env.target_tangent[1,: ], env.target_tangent[ 2,:], color='black', length=0.3)
    ax1.set_title('Target TCP Tip\n(pos + orient in base frame)')
    ax1.set_xlabel('x_t')
    ax1.set_ylabel('y_t')
    ax1.set_zlabel('z_t')
    ax1.set_xlim(247, 257)
    ax1.set_ylim(-253, -244)
    ax1.set_zlim(70.0, 70.4)
    ax1.set_box_aspect((10, 4, 1))
    #ax1.set_box_aspect((1,1,1))
    #ax1.view_init(elev=20, azim=-80)
    #ax1.view_init(elev=90, azim=0) # Emulates view(90,0)
    ax1.grid(True)

    # draw_box(ax1, env.world_box_center_base[0], env.world_box_center_base[1], env.world_box_center_base[2], L=0.06*1e3, color='orange')
    ax1.scatter(env.ellipse_center_base[0], env.ellipse_center_base[1], env.ellipse_center_base[2], color="red", s=30, label="ellipse center")
    ax1.scatter(p[0], p[1], p[2], color="green", s=30, label="start point")
    # p = env.ellipse_center_base#T_global2base@T_t2tcp@(np.concatenate([np.array([[-289.997], [+6.666], [+297.00]]).flatten(),[1]]))
    # ax1.scatter(p[0], p[1], p[2], color="black", s=30, label="Target")
    # p = env.world_box_center_base#T_global2base@T_t2tcp@(np.concatenate([np.array([[-289.997], [+6.666], [+297.00]]).flatten(),[1]]))
    # ax1.scatter(p[0], p[1], p[2], color="red", s=30, label="Target")
    ax1.legend()
    #Subplot 2: Cross Error Vector
    ax2 = fig.add_subplot(1, 3, 3)#, projection='3d')
    ax2.plot(list(range(len(cross_error_vec))), cross_error_vec, 'b', linewidth=2)
    ax2.scatter(list(range(len(cross_error_vec))), cross_error_vec, c='b', linewidth=2)
    ax2.plot(list(range(len(cross_error_vec))), [0.05]*len(cross_error_vec), 'r--', linewidth=1)
    for i in range(len(cross_error_vec)):
        ax2.text(i, cross_error_vec[i], f'{cross_error_vec[i]:.2f}', fontsize=8)
    #ax2.text(list(range(len(cross_error_vec))), cross_error_vec,[f'{x:.2f}' for x in cross_error_vec])
    #ax2.quiver(P_tcp[0, :], P_tcp[1, :], P_tcp[2, :], O_tcp[0, :], O_tcp[1, :], O_tcp[2, :], color='r', length=0.2)
    ax2.set_title('TCP cross error\n(base frame)')
    ax2.set_xlabel('steps')
    ax2.set_ylabel('error(mm)')
    #ax2.set_zlabel('z_tcp')
    ax2.grid(True)

    # # Subplot 3: Required Base Position
    ax3 = fig.add_subplot(1, 3, 2)#, projection='3d')
    ax3.plot(list(range(len(theta_list))), theta_list, 'b', linewidth=2)
    ax3.scatter(list(range(len(theta_list))), theta_list, c='b', linewidth=2)
    ax3.plot(list(range(len(theta_list))), [1.5]*len(theta_list), 'r--', linewidth=1)
    for i in range(len(theta_list)):
        ax3.text(i, theta_list[i], f'{theta_list[i]:.2f}', fontsize=8)
    #ax3.text(list(range(len(theta_list))), theta_list,[f'{x:.2f}' for x in theta_list])
    #ax2.quiver(P_tcp[0, :], P_tcp[1, :], P_tcp[2, :], O_tcp[0, :], O_tcp[1, :], O_tcp[2, :], color='r', length=0.2)
    ax3.set_title('TCP orientation error\n(base frame)')
    ax3.set_xlabel('steps')
    ax3.set_ylabel('error(°)')
    #ax3.set_zlabel('z_tcp')
    ax3.grid(True)



    positions.append(position.copy())
    orientations.append(orientation.copy())
    cross_error_vecs.append(cross_error_vec.copy())
    ori_errors.append(ori_error.copy())
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
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure(figsize=(18, 6))
#positions = #[np.array([ 240.18, -243.96,   66.58]), np.array([ 241.03, -244.4 ,   67.4 ]), np.array([ 251.54, -242.12,   74.57]), np.array([ 251.49, -242.08,   74.89]), np.array([ 245.88, -240.06,   67.53]), np.array([ 246.56, -241.35,   68.41]), np.array([ 249.17, -242.81,   72.98]), np.array([ 247.45, -243.74,   71.88]), np.array([ 246.3 , -245.03,   70.76]), np.array([ 246.93, -245.6 ,   70.4 ]), np.array([ 246.98, -245.96,   70.02]), np.array([ 246.99, -245.98,   70.01]), np.array([ 247.27, -246.77,   70.03]), np.array([ 247.17, -246.84,   70.08]), np.array([ 247.15, -246.87,   70.01]), np.array([ 247.16, -246.84,   70.04]), np.array([ 247.17, -246.84,   70.01]), np.array([ 247.18, -246.82,   70.03]), np.array([ 247.16, -246.82,   70.02]), np.array([ 247.13, -246.79,   70.  ]), np.array([ 247.13, -246.78,   69.99]), np.array([ 247.14, -246.77,   70.  ]), np.array([ 247.14, -246.76,   70.  ]), np.array([ 247.17, -246.77,   70.02]), np.array([ 247.16, -246.75,   70.01]), np.array([ 247.18, -246.76,   70.  ]), np.array([ 247.16, -246.76,   69.99]), np.array([ 247.13, -246.75,   69.98]), np.array([ 247.13, -246.74,   69.95]), np.array([ 247.15, -246.78,   69.96]), np.array([ 247.18, -246.78,   70.  ]), np.array([ 247.17, -246.79,   70.  ]), np.array([ 247.17, -246.81,   69.99]), np.array([ 247.17, -246.8 ,   69.99]), np.array([ 247.66, -247.45,   69.97]), np.array([ 247.59, -247.46,   70.03]), np.array([ 247.62, -247.48,   70.02]), np.array([ 247.63, -247.46,   70.04]), np.array([ 247.63, -247.47,   70.03]), np.array([ 247.64, -247.46,   70.04]), np.array([ 247.66, -247.47,   70.03]), np.array([ 247.66, -247.48,   70.03]), np.array([ 247.66, -247.46,   70.03]), np.array([ 247.64, -247.47,   70.02]), np.array([ 247.65, -247.45,   70.  ]), np.array([ 247.63, -247.46,   69.98]), np.array([ 247.63, -247.48,   70.  ]), np.array([ 247.65, -247.48,   70.  ]), np.array([ 247.66, -247.5 ,   69.99]), np.array([ 247.64, -247.49,   69.98]), np.array([ 248.29, -248.02,   70.03]), np.array([ 248.27, -248.  ,   70.05]), np.array([ 248.27, -248.  ,   70.05]), np.array([ 248.29, -248.  ,   70.05]), np.array([ 248.28, -248.  ,   70.04]), np.array([ 248.27, -248.  ,   70.03]), np.array([ 248.28, -247.98,   70.01]), np.array([ 248.28, -248.  ,   70.01]), np.array([ 248.27, -248.  ,   70.  ]), np.array([ 248.28, -248.  ,   70.  ]), np.array([ 248.25, -248.01,   69.99]), np.array([ 248.95, -248.37,   70.01]), np.array([ 248.92, -248.38,   70.02]), np.array([ 248.94, -248.36,   70.02]), np.array([ 248.93, -248.36,   70.01]), np.array([ 248.95, -248.38,   70.01]), np.array([ 248.94, -248.36,   70.01]), np.array([ 248.93, -248.38,   70.  ]), np.array([ 248.93, -248.38,   70.  ]), np.array([ 249.56, -248.64,   69.96]), np.array([ 249.58, -248.65,   69.97]), np.array([ 249.58, -248.62,   70.01]), np.array([ 249.58, -248.61,   70.01]), np.array([ 249.57, -248.62,   70.  ]), np.array([ 250.19, -248.79,   69.98]), np.array([ 250.19, -248.81,   69.99]), np.array([ 250.17, -248.78,   70.  ]), np.array([ 250.17, -248.79,   70.  ]), np.array([ 250.18, -248.78,   70.  ]), np.array([ 250.75, -248.9 ,   70.  ]), np.array([ 250.73, -248.89,   70.  ]), np.array([ 250.73, -248.89,   70.  ]), np.array([ 250.71, -248.88,   70.  ]), np.array([ 251.25, -248.94,   70.03]), np.array([ 251.24, -248.96,   70.  ]), np.array([ 251.22, -248.99,   69.98]), np.array([ 251.77, -248.98,   70.02]), np.array([ 251.78, -249.01,   70.  ]), np.array([ 251.74, -248.99,   70.01]), np.array([ 252.25, -248.98,   70.  ]), np.array([ 252.25, -248.99,   70.01]), np.array([ 252.25, -248.97,   70.  ]), np.array([ 252.76, -248.95,   70.01]), np.array([ 252.76, -248.95,   70.  ]), np.array([ 252.75, -248.96,   70.  ]), np.array([ 253.28, -248.88,   70.  ]), np.array([ 253.27, -248.89,   70.  ]), np.array([ 253.27, -248.88,   70.  ]), np.array([ 253.25, -248.88,   70.  ]), np.array([ 253.81, -248.81,   70.  ]), np.array([ 253.81, -248.77,   70.  ]), np.array([ 253.83, -248.77,   70.  ]), np.array([ 253.8 , -248.77,   70.01]), np.array([ 254.41, -248.67,   69.99]), np.array([ 254.41, -248.61,   70.  ]), np.array([ 254.42, -248.62,   70.01]), np.array([ 254.4 , -248.62,   69.99]), np.array([ 255.04, -248.4 ,   69.96]), np.array([ 255.03, -248.38,   69.97]), np.array([ 255.05, -248.38,   69.97]), np.array([ 255.06, -248.37,   70.  ]), np.array([ 255.05, -248.38,   70.  ]), np.array([ 255.1 , -248.4 ,   69.99]), np.array([ 255.69, -248.02,   70.02]), np.array([ 255.66, -248.  ,   70.  ]), np.array([ 255.68, -248.  ,   69.99]), np.array([ 255.68, -248.01,   69.99]), np.array([ 255.71, -247.99,   69.99]), np.array([ 255.71, -248.  ,   70.01]), np.array([ 255.7 , -247.99,   69.99]), np.array([ 255.73, -248.01,   70.02]), np.array([ 256.32, -247.5 ,   70.03]), np.array([ 256.3 , -247.5 ,   70.02]), np.array([ 256.3 , -247.49,   70.02]), np.array([ 256.3 , -247.49,   70.01]), np.array([ 256.31, -247.48,   69.99]), np.array([ 256.31, -247.48,   70.  ]), np.array([ 256.32, -247.46,   70.  ]), np.array([ 256.33, -247.48,   70.  ]), np.array([ 256.34, -247.46,   70.01]), np.array([ 256.34, -247.48,   70.  ]), np.array([ 256.33, -247.47,   69.99]), np.array([ 256.53, -247.39,   70.06]), np.array([ 256.82, -246.83,   70.03]), np.array([ 256.82, -246.82,   70.02]), np.array([ 256.82, -246.81,   70.03]), np.array([ 256.82, -246.82,   70.02]), np.array([ 256.81, -246.82,   70.02]), np.array([ 256.82, -246.82,   70.02]), np.array([ 256.81, -246.82,   70.01]), np.array([ 256.82, -246.8 ,   70.02]), np.array([ 256.83, -246.8 ,   70.  ]), np.array([ 256.83, -246.79,   70.01]), np.array([ 256.83, -246.78,   70.01]), np.array([ 256.83, -246.78,   70.01]), np.array([ 256.84, -246.78,   70.01]), np.array([ 256.85, -246.79,   70.01]), np.array([ 256.85, -246.81,   70.01]), np.array([ 256.8 , -246.77,   69.99]), np.array([ 257.07, -246.69,   70.05]), np.array([ 256.82, -246.79,   70.  ]), np.array([ 256.83, -246.79,   69.92]), np.array([ 256.79, -246.77,   70.  ]), np.array([ 256.84, -246.79,   69.94]), np.array([ 257.04, -245.96,   70.03]), np.array([ 257.04, -245.99,   70.03]), np.array([ 257.03, -246.01,   70.04]), np.array([ 257.02, -246.  ,   70.02]), np.array([ 257.03, -246.  ,   70.03]), np.array([ 257.03, -246.  ,   70.03]), np.array([ 257.02, -246.  ,   70.02]), np.array([ 257.02, -246.01,   70.02]), np.array([ 257.02, -246.01,   70.02]), np.array([ 257.02, -246.01,   70.01]), np.array([ 257.02, -246.01,   70.02]), np.array([ 257.03, -246.  ,   70.01]), np.array([ 257.04, -246.  ,   70.01]), np.array([ 257.04, -245.99,   70.01]), np.array([ 257.04, -245.99,   70.02]), np.array([ 257.03, -245.98,   70.02]), np.array([ 257.04, -245.98,   70.02]), np.array([ 257.05, -245.99,   70.02]), np.array([ 257.05, -245.99,   70.02]), np.array([ 257.05, -246.03,   70.02]), np.array([ 257.06, -246.05,   70.02]), np.array([ 257.05, -246.06,   70.03]), np.array([ 257.05, -246.07,   70.04]), np.array([ 257.04, -246.05,   70.04]), np.array([ 257.05, -246.05,   70.03]), np.array([ 257.05, -246.05,   70.03]), np.array([ 257.05, -246.06,   70.04]), np.array([ 257.04, -246.07,   70.03]), np.array([ 257.05, -246.06,   70.04]), np.array([ 257.04, -246.07,   70.04]), np.array([ 257.03, -246.07,   70.03]), np.array([ 257.04, -246.08,   70.03]), np.array([ 257.03, -246.08,   70.04]), np.array([ 257.03, -246.09,   70.03]), np.array([ 257.03, -246.09,   70.04]), np.array([ 257.02, -246.1 ,   70.04]), np.array([ 257.03, -246.1 ,   70.03]), np.array([ 257.02, -246.11,   70.04]), np.array([ 257.02, -246.12,   70.04]), np.array([ 257.02, -246.12,   70.05]), np.array([ 257.02, -246.12,   70.04]), np.array([ 257.01, -246.12,   70.04]), np.array([ 257.02, -246.12,   70.05]), np.array([ 257.03, -246.12,   70.04]), np.array([ 257.02, -246.13,   70.05]), np.array([ 257.02, -246.12,   70.05]), np.array([ 257.03, -246.13,   70.05]), np.array([ 257.02, -246.13,   70.05]), np.array([ 257.03, -246.13,   70.05]), np.array([ 257.02, -246.13,   70.05]), np.array([ 257.03, -246.13,   70.05]), np.array([ 257.03, -246.13,   70.05]), np.array([ 257.03, -246.13,   70.06]), np.array([ 257.03, -246.13,   70.05]), np.array([ 257.03, -246.13,   70.06]), np.array([ 257.03, -246.13,   70.05]), np.array([ 257.03, -246.13,   70.06]), np.array([ 257.04, -246.13,   70.05]), np.array([ 257.04, -246.13,   70.06]), np.array([ 257.04, -246.14,   70.06]), np.array([ 257.03, -246.13,   70.06]), np.array([ 257.04, -246.13,   70.06]), np.array([ 257.04, -246.13,   70.06]), np.array([ 257.05, -246.13,   70.06]), np.array([ 257.04, -246.13,   70.06]), np.array([ 257.04, -246.13,   70.07]), np.array([ 257.05, -246.13,   70.06]), np.array([ 257.04, -246.12,   70.06]), np.array([ 257.05, -246.13,   70.07]), np.array([ 257.05, -246.13,   70.06]), np.array([ 257.05, -246.12,   70.06]), np.array([ 257.04, -246.13,   70.07]), np.array([ 257.06, -246.13,   70.06]), np.array([ 257.05, -246.12,   70.07]), np.array([ 257.05, -246.13,   70.07]), np.array([ 257.05, -246.12,   70.06]), np.array([ 257.05, -246.12,   70.07]), np.array([ 257.05, -246.13,   70.07]), np.array([ 257.06, -246.12,   70.07]), np.array([ 257.05, -246.13,   70.07]), np.array([ 257.06, -246.12,   70.07]), np.array([ 257.05, -246.12,   70.07]), np.array([ 257.06, -246.12,   70.07]), np.array([ 257.06, -246.13,   70.07]), np.array([ 257.05, -246.12,   70.07]), np.array([ 257.06, -246.12,   70.07]), np.array([ 257.06, -246.12,   70.08]), np.array([ 257.05, -246.12,   70.07]), np.array([ 257.06, -246.12,   70.07]), np.array([ 257.06, -246.12,   70.08]), np.array([ 257.06, -246.12,   70.07]), np.array([ 257.06, -246.12,   70.08]), np.array([ 257.06, -246.12,   70.07]), np.array([ 257.06, -246.12,   70.08]), np.array([ 257.06, -246.12,   70.07]), np.array([ 257.07, -246.12,   70.08]), np.array([ 257.06, -246.12,   70.08]), np.array([ 257.06, -246.11,   70.08]), np.array([ 257.07, -246.12,   70.07]), np.array([ 257.05, -246.12,   70.08]), np.array([ 257.07, -246.12,   70.08]), np.array([ 257.06, -246.11,   70.08]), np.array([ 257.07, -246.12,   70.08]), np.array([ 257.06, -246.12,   70.08]), np.array([ 247.16, -250.63,   72.63]), np.array([ 246.76, -249.12,   71.56])]
#position = np.array([a.tolist() for a in position])
#O_t = np.array([orientation[:, 0] for orientation in orientations]).T
# Subplot 1: Target Needle Tip
# ax1 = fig.add_subplot(1, 2, 1, projection='3d')
# ax1.plot3D(position[:, 0], position[:, 1], position[:, 2], 'b', linewidth=2)
# ax1.scatter3D(position[:, 0], position[:, 1], position[:, 2], 'b', linewidth=2)
# for i in range(len(position)):
#     ax1.text(position[i, 0], position[i, 1], position[i, 2], str(i), zdir='z', fontsize=8)
# #ax1.text3D(positions[:, 0], positions[:, 1], positions[:, 2], [str(i) for i in range(len(positions))], zdir='z', fontsize=8)
# ax1.plot3D(env.target_path[0, :], env.target_path[1,: ], env.target_path[ 2,:], 'g', linewidth=2)
# ax1.scatter3D(env.target_path[0, :], env.target_path[1,: ], env.target_path[ 2,:], 'g', linewidth=2)
# #ax1.quiver(positions[:, 0], positions[:, 1], positions[:, 2], O_t[0, :], O_t[1, :], O_t[2, :], color='r', length=0.3)
# ax1.set_title('Target Needle Tip\n(global pos + global orient)')
# ax1.set_xlabel('x_t')
# ax1.set_ylabel('y_t')
# ax1.set_zlabel('z_t')
# ax1.set_xlim(247, 257)
# ax1.set_ylim(-253, -244)
# ax1.set_zlim(70.0, 70.4)

# ax1.set_box_aspect((10, 4, 0.4))
# ax1.view_init(elev=20, azim=-80)
# #ax1.view_init(elev=90, azim=0) # Emulates view(90,0)
# ax1.grid(True)
print(cross_error_vecs[0])
ax2 = fig.add_subplot(1, 1, 1)#, projection='3d')
ax2.plot(list(range(max([len(vec) for vec in cross_error_vecs]))), [0.05]*max([len(vec) for vec in cross_error_vecs]), 'r--', linewidth=3)
for i, cross_error_vec in enumerate(cross_error_vecs):
    #ax2.scatter(list(range(len(cross_error_vec))), cross_error_vec, linewidth=1)
    ax2.plot(list(range(len(cross_error_vec))), cross_error_vec, linewidth=1, label=f'eps {i+1}')
    
    # for j in range(len(cross_error_vec)):
    #     ax2.text(j, cross_error_vec[j], f'{cross_error_vec[j]:.2f}', fontsize=8)
    #ax2.text(list(range(len(cross_error_vec))), cross_error_vec,[f'{x:.2f}' for x in cross_error_vec])
    #ax2.quiver(P_tcp[0, :], P_tcp[1, :], P_tcp[2, :], O_tcp[0, :], O_tcp[1, :], O_tcp[2, :], color='r', length=0.2)
ax2.set_title('Target tcp cross error\n(base position)')
ax2.set_xlabel('steps')
ax2.set_ylabel('error(mm)')
#ax2.set_zlabel('z_tcp')
ax2.grid(True)
# #Subplot 2: Cross Error Vector
# ax2 = fig.add_subplot(1, 2, 2)#, projection='3d')
# ax2.plot(list(range(len(cross_error_vec))), cross_error_vec, 'b', linewidth=2)
# ax2.scatter(list(range(len(cross_error_vec))), cross_error_vec, c='b', linewidth=2)
# ax2.plot(list(range(len(cross_error_vec))), [0.05]*len(cross_error_vec), 'r--', linewidth=1)
# for i in range(len(cross_error_vec)):
#     ax2.text(i, cross_error_vec[i], f'{cross_error_vec[i]:.2f}', fontsize=8)
# #ax2.text(list(range(len(cross_error_vec))), cross_error_vec,[f'{x:.2f}' for x in cross_error_vec])
# #ax2.quiver(P_tcp[0, :], P_tcp[1, :], P_tcp[2, :], O_tcp[0, :], O_tcp[1, :], O_tcp[2, :], color='r', length=0.2)
# ax2.set_title('Target tcp cross error\n(base position)')
# ax2.set_xlabel('steps')
# ax2.set_ylabel('error(mm)')
# #ax2.set_zlabel('z_tcp')
# ax2.grid(True)
plt.tight_layout()
plt.ion()
plt.legend()
plt.show(block=True)    
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
