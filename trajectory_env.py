
import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import os
from trajectory import elliptical_trajectory, T_global2base, elipse_center_global
from gymnasium.utils.env_checker import check_env


def clean_null_noise(arr, null_threshold=1e-8):
    return np.where(np.abs(arr) < null_threshold, 0.0, arr)
# radians angles are rounded to 4 decimal places to avoid numerical issues, and positions are rounded to 2 decimal places to avoid numerical issues
# orientations matrices/vectors are rounded to 3 decimal places to avoid numerical issues
# you can completely modify this class for your MuJoCo environment by following the directions
class ArmRobot(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ]#,
        #"render_fps": 100,
    }

    # set default episode_len for truncate episodes 
    # eps_pos and eps_ori are the thresholds for position and orientation errors to consider the target reached
    # eps_pos is in meters and eps_ori is in degrees
    # max_joint_delta represents the max allowed angle adjustement per step and is in in degrees
    def __init__(self,null_threshold=1e-8, work_box_size=0.1, threshold_pos=500e-6, threshold_ori=1.50, start_target_idx=0, max_joint_delta = 3.0, render_mode=None, site_name="needle_holder_site", episode_len=500, num_points=19, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        # change shape of observation to your observation space size
        observation_space = Box(low=-np.inf, high=np.inf, shape=(48,), dtype=np.float64)

        self.ellipse_center_base = T_global2base@(np.concatenate([np.array(elipse_center_global).flatten(),[1]]))
        self.ellipse_center_base = self.ellipse_center_base[:3] # in mm
        half_extents = np.array([work_box_size/2, work_box_size/2, work_box_size/2], dtype=np.float64) * 1e3 # convert in mm

        # work space
        self.work_space = Box(
            low=self.ellipse_center_base - half_extents,
            high=self.ellipse_center_base + half_extents,
            dtype=np.float64
        )


        self.null_threshold = null_threshold
        self.threshold_pos = threshold_pos*1e3 # convert to mm
        self.threshold_ori = np.round(np.deg2rad(threshold_ori), 4) # convert to radians
        self.xml_needle_holder_name = site_name
        self.num_points = num_points
        # # world space, reduced3D space representing a reduction of the workspace of the robot
        # self.world_space = Box(low=np.array([-1.0, -1.0, 0.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float64)

        # load your MJCF model with env and choose frames count between actions
        MujocoEnv.__init__(
            self,
            model_path=os.path.abspath("universal_robots_ur3e-main/scene.xml"),
            frame_skip=5,
            observation_space=observation_space,
            render_mode=render_mode,
            **kwargs
        )
        self.viewer = None
        self.metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": int(np.round(1.0 / self.dt)),
    }

        bounds = self.model.actuator_ctrlrange.copy().astype(np.float64)
        self.actions_low, self.actions_high = bounds.T
        num_actions = len(self.actions_low)
        self._max_joint_delta = max_joint_delta # in degrees, maximum change in joint angles per step
        self.needle_holder_site_id = self.model.site(self.xml_needle_holder_name).id
        # Override the action space for the RL agent
        delta_bounds = np.ones(num_actions, dtype=np.float64) 
        self.action_space = Box(
            low=-delta_bounds, 
            high=delta_bounds, 
            dtype=np.float64
        )

        self.step_number = 0
        # self.steps_to_pos_current_target = 0
        # self.steps_to_ori_current_target = 0
        self.episode_len = episode_len
        self.path_start_idx = start_target_idx
        #self.init_qpos = np.zeros(self.model.nq)#self.model.key_qpos[0].copy() # joint positions of the robot at the beginning of the episode(first point of the trajectory)
        
        #Init joints in degrees and convert to radians from autonomous_surgery repo 
        #TODO: fine tune to match the pose for the first point of the trajectory
        init_joints_degrees = [-21.46, -97.78, -66.61, -106.15, 88.77, -21.62] #[-10.042, -93.752, -63.163, -109.568, 76.181, -11.005]
        #init_joints_degrees = [-21.58, -97.97, -66.38, -106.18, 88.76, -21.74]
        # init_joints_grad = [angle * np.pi / 180 for angle in init_joints_degrees]
        self.init_qpos = np.round(np.deg2rad(init_joints_degrees), 4) #self.model.keyframe("home").qpos.copy() # joint positions of the robot at the beginning of the episode(first point of the trajectory)

        # precomputed target trajectory and orientation of the TCP given in the base frame in mm
        _ , trajectory_in_base_frame = elliptical_trajectory(a=5, b=3, N=self.num_points, show_plot=False)
        self.target_path, self.target_tangent, self.target_binormal, self.target_normal = trajectory_in_base_frame
        print(self.target_path)
        self.target_path = np.round(clean_null_noise(self.target_path, self.null_threshold), 2) # in mm
        self.target_tangent = np.round(clean_null_noise(self.target_tangent, self.null_threshold), 3)
        self.target_binormal = np.round(clean_null_noise(self.target_binormal, self.null_threshold), 3)
        self.target_normal = np.round(clean_null_noise(self.target_normal, self.null_threshold), 3)

        # will be set at reset time to a point near the first point of the trajectory( for the moment,
        # it will be set to the first trajectory point later, but it can be set to a random point near the first trajectory point)
        # self.current_pos = None # np.array([self.target_path[0,0], self.target_path[1,0], self.target_path[2,0]])
        # self.current_ori = None # np.array([self.target_orientations[0,0], self.target_orientations[1,0], self.target_orientations[2,0]])

        # self.p_err = None
        # self.ori_err = None
        self.w_pos = 0.59 #4 #0.55 #0.55
        self.w_ori = 0.40 #3 #0.44 #0.35
        self.w_cost = 0.01 #0.10 weight for the control cost in the reward function
        self.visited_points = [False] * self.num_points #len(self.target_path[0])
        self.visited_orientations = [False] * self.num_points #len(self.target_path[0])
        #self.visited_points[0] = True

        # self.target_pos_index = self.path_start_idx
        # self.target_ori_index = self.path_start_idx
        self.current_target_idx = self.path_start_idx
        
    @property
    def current_target_pos(self):
        # shape 3 x 1 in mm
        return np.array(self.target_path[:,self.current_target_idx])
    
    def get_target_pos_at(self, idx):
        # shape 3 x 1 in mm
        return np.array(self.target_path[:,idx])
    
    @property
    def current_target_ori(self):
        # shape 3 x 3
        return  np.array([self.target_tangent[:,self.current_target_idx],
                         self.target_binormal[:,self.current_target_idx],
                         self.target_normal[:,self.current_target_idx]]).T

    def get_target_ori_at(self, idx):
        # shape 3 x 3
        return  np.array([self.target_tangent[:,idx],
                         self.target_binormal[:,idx],
                         self.target_normal[:,idx]]).T
    #def get_p
    # determine the reward depending on observation or other properties of the simulation
    def step(self, d_action):

        # fetch current control value
        current_ctrl = self.data.qpos.copy()

        # scale the action to the max allowed joint delta and convert to radians
        d_action = (np.clip(d_action, -1.0, 1.0) * self._max_joint_delta) # convert to radians
        d_action = np.deg2rad(d_action) # convert to radians
        # add the delta. Noise will be handle by SB3 Wrapper during training. It will be added to d_action before passing it to / calling "step()"
        absolute_action = current_ctrl + d_action

        # Apply delta and clip to XML ctrlrange boundaries
        clipped_action = np.clip(
            absolute_action, 
            self.actions_low, 
            self.actions_high
        )
        clipped_action = np.round(clipped_action, 4) # round to 4 decimal places to avoid numerical issues
        # perform the simulation step with the given action and frame_skip 
        self.do_simulation(clipped_action, self.frame_skip)

        # step counter
        self.step_number += 1

        # update the number of steps taken to reach the current target position and orientation
        # until the target is reached, if the target is reached, the counter will not be updated anymore until the next target is set
        # self.steps_to_pos_current_target += 1*(not self.visited_points[self.current_target_idx])
        # self.steps_to_ori_current_target += 1*(not self.visited_orientations[self.current_target_idx])

        self.update_visited_checkpoints()
        # get the observation after the simulation step
        obs = self._get_obs()

        # compute the reward and the termination conditions
        reward, rew_info = self._get_reward(d_action)

        
        # self.update_pos_ori_weights()
        terminated = self.check_termination_conditions()
        truncated = self.step_number > self.episode_len
        info = {
            "d_action": np.round(d_action, 4),
            "action": clipped_action,
            "step_number": self.step_number,
            # "steps_to_pos_current_target": self.steps_to_pos_current_target,
            # "steps_to_ori_current_target": self.steps_to_ori_current_target,
            "current_target_idx": self.current_target_idx,
            "current_pos": self.current_pos,
            "current_ori": self.current_ori,
            "current_target_pos": self.current_target_pos,
            "current_target_ori": self.current_target_ori,
            "is_last_target": self.is_last_target(),
            "pos_err": self.pos_err,
            "ori_err": self.ori_err,
            "visited_points": self.visited_points,
            "visited_orientations": self.visited_orientations,
            **rew_info
        }
        #print(info)
        print(f"Step: {self.step_number}, Target idx: {self.current_target_idx}, Pos err: {self.pos_err:.6f}, Ori err: {self.ori_err:.6f}, Reward: {reward:.6f}")
        print(f"err_mat_trace: {np.trace(self.ori_err_mat)}, d_action: {d_action}, action: {clipped_action}")
        print(f"current_pos: {self.current_pos}, target_pos: {self.current_target_pos}")#, current_ori: {self.current_ori}, target_ori: {self.current_target_ori}")
        print(f"current_ori: \n{self.current_ori}, \ntarget_ori: \n{self.current_target_ori} \nerror_mat: \n{self.ori_err_mat}")
        print(f"visited_points: {self.visited_points}")
        print(f"visited_orientations: {self.visited_orientations}")
        if self.render_mode == "human":
            self.render()
        
        self.update_current_target()
        return obs, reward, terminated, truncated, info

    def update_visited_checkpoints(self):
        if self.ori_err < self.threshold_ori:
            self.visited_orientations[self.current_target_idx] = True
        else:
            self.visited_orientations[self.current_target_idx] = False

        if self.pos_err < self.threshold_pos:
            self.visited_points[self.current_target_idx] = True
        else:
            self.visited_points[self.current_target_idx] = False

    # def update_pos_ori_weights(self):
    #     if self.visited_points[self.current_target_idx] ^ self.visited_orientations[self.current_target_idx]:
    #         self.w_pos, self.w_ori = (0.3, 0.7) if self.visited_points[self.current_target_idx] else (0.7, 0.3)
    #     else:
    #         self.w_pos, self.w_ori = (0.5, 0.5)

    def current_target_reached(self):
        return self.visited_points[self.current_target_idx] and self.visited_orientations[self.current_target_idx]

    @property
    def bonus_current_target_reached(self):
        # return a bonus reward if the current target is reached, with a higher bonus if both position and orientation are reached at the same time step
        bonus = 0.0
        if self.current_target_reached():
            bonus = 1.0 #if self.steps_to_ori_current_target == self.steps_to_pos_current_target else 0.5 # bonus for reaching the target
            if self.is_last_target():
                bonus += 10.0 # additional bonus for reaching the last target
        return bonus
    
    @property
    def penalty_leaving_workspace(self):
        # return a penalty if the robot leaves the workspace
        penalty = 0.0
        if not self.work_space.contains(self.current_pos) or self.pos_err > 4:
            penalty = -5.0 #if self.steps_to_pos_current_target > 10 else -0.5 # penalty for leaving the workspace
        return penalty

    #TODO: Add a penalty if the robot is too far from the target position or orientation for too long
    # @property
    # def penalty_too_high_pos_ori_err(self):
    #     # return a penalty if the position or orientation error is too high
    #     penalty = 0.0
    #     if self.pos_err > 10*self.threshold_pos or self.ori_err > 10*self.threshold_ori:
    #         penalty = - np.exp(self.pos_err + self.ori_err) # penalty for too high position or orientation error 
    #     return penalty
    
    def update_current_target(self):
        if self.current_target_reached():
            # update the target to the next point in the trajectory
            if self.current_target_idx < self.num_points - 1:
                self.current_target_idx += 1
                # self.w_pos, self.w_ori = 0.5, 0.5
                # self.steps_to_pos_current_target = 0
                # self.steps_to_ori_current_target = 0
            # else:
            #     # if the last point of the trajectory is reached, the episode can be terminated
            #     pass

    # def last_target_reached(self):
    #     return self.current_target_idx == (len(self.target_path[0]) - 1) and self.current_target_reached()
    
    def is_last_target(self):
        return self.current_target_idx == (self.num_points - 1)
    
    def check_termination_conditions(self):
        # check termination conditions, 
        # for example if the end effector is too far from the target or if the episode length is reached
        # or if the robot reaches the target position and orientation, then update the target to the next point in the trajectory
        terminated = False
        if not self.work_space.contains(self.current_pos) or self.pos_err > 4:
            terminated = True
        # elif self.step_number > self.episode_len:
        #     terminated = True
        elif self.is_last_target() and self.current_target_reached(): #last target reached
            terminated = True
        #TODO: Add a termination condition if the robot is too far from the target position or orientation for too long
        return terminated  


    def _get_crtl_cost(self, d_action):
        # compute the control cost based on the action taken
        # for example, the control cost can be computed as the L2 norm of the action
        ctrl_cost = np.sum(np.square(d_action))
        return ctrl_cost
    
    def _get_position_reward(self):
        #sigma_p = self.threshold_pos
        # r_pos = 0.5 * np.exp( (self.pos_err ** 2) / ((10*sigma_p)** 2)) + 0.5 * np.exp( (self.pos_err ** 2) / ( sigma_p** 2)) # Gaussain RBF, radial basis function
        # r_pos = np.exp(-(self.pos_err/(10*self.threshold_pos))) + np.exp(-(self.pos_err/self.threshold_pos)) + np.exp(-((10*self.pos_err)/self.threshold_pos)) #+ np.exp(-(self.pos_err/self.threshold_pos)**2) # Gaussian RBF, radial basis function
        r_pos = 10*np.exp(-(self.pos_err/self.threshold_pos))
        return np.round(r_pos, 4)
    
    def _get_orientation_reward(self):
        # o_err_mat = self._get_ori_err_mat()
        # trace_val = np.sum(self.current_ori * self.current_target_ori) # trace with Frobenius inner product
        # cos_theta = np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0)
        # self.ori_err = np.arccos(cos_theta)
        # assert self.ori_err == np.arccos(np.clip((np.trace(self.ori_err_mat) - 1.0) / 2.0, -1.0, 1.0)), "Both Frobenius inner product and np.trace() must produce the same result. check the values and comment the assert if the error is due to rounding errors"
        ori_cost = 0.0
        if self.ori_err_mat[0,0] < 0 or self.ori_err_mat[1,1] < 0 or self.ori_err_mat[2,2] < 0:
            ori_cost = -1.0 # penalty for negative diagonal elements in the orientation error matrix
        mat_trace = np.trace(self.ori_err_mat) #(np.trace(self.ori_err_mat) + 1) / 4
        # 2. Shifted exponential (k = 2)
        k = 50
        r_ori = 10 * ((np.exp(k * (mat_trace + 1) / 4) - 1) / (np.exp(k) - 1)) # 3 * (np.exp(k * (mat_trace + 1) / 4) - 1) / (np.exp(k) - 1)
        # r_ori = np.exp(-(self.ori_err/(10*self.threshold_ori))) + np.exp(-(self.ori_err/self.threshold_ori)) + np.exp(-((10*self.ori_err)/self.threshold_ori)) #+ np.exp(-(self.ori_err/self.threshold_ori)**2)
        #r_ori = - 1/np.exp(r_ori) # Gaussian RBF, radial basis function
        return np.round(r_ori + ori_cost, 4)

    def _get_timestep_reward(self):
        return -1.0 # constant penalty for each timestep to encourage faster completion of the task
    
    def _get_reward(self, d_action):

        # R POSITION compute position error reward
        r_pos = self._get_position_reward()#obs[:3])
        # R ORIENTATION compute orientation reward
        r_ori = self._get_orientation_reward()
        # cost JOINT VELOCITY PENALTY
        c_vel = -self._get_crtl_cost(d_action)

        r_timestep = self._get_timestep_reward()

        bonus = self.bonus_current_target_reached
        penalty = self.penalty_leaving_workspace


        reward = self.w_pos*r_pos + self.w_ori*r_ori + self.w_cost*c_vel + r_timestep + bonus + penalty  #... # compute the reward based on the observation and other properties of the simulation
        #TODO: Add a penalty if the robot is too far from the target position or orientation for too long
        reward_info = {
            "reward_position": r_pos,
            "reward_orientation": r_ori,
            "reward_control_cost": c_vel,
            "bonus_current_target_reached": bonus,
            "penalty_leaving_workspace": penalty,
        }
        return reward, reward_info

    # define what should happen when the model is reset (at the beginning of each episode)
    def reset_model(self):
        # TODO: complete the reset_model function to reset the robot to its initial state and set the target to the first point in the trajectory
        self.step_number = 0
        #self.w_pos, self.w_ori = (0.5, 0.5)
        # self.steps_to_pos_current_target = 0
        # self.steps_to_ori_current_target = 0

        self.current_target_idx = self.path_start_idx
        self.visited_points = [False] * self.num_points
        self.visited_orientations = [False] * self.num_points
        # self.target_ori_index = self.path_start_idx
        # self.current_pos = np.array([self.target_path[0,0], self.target_path[1,0], self.target_path[2,0]])
        # self.current_ori = np.array([self.target_tangent[0,0], self.target_tangent[1,0], self.target_tangent[2,0]])

        # self.model.keyframe('keyframe_name').id
        # Clears the slate and loads the XML's keyframe 0 configuration
        # mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        print(self.target_path)
        # for example, noise is added to positions and velocities
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )
        self.set_state(qpos, qvel)
        return self._get_obs()


    def _get_reset_info(self):
        reset_info = {
            "step_number": self.step_number,
            # "steps_to_pos_current_target": self.steps_to_pos_current_target,
            # "steps_to_ori_current_target": self.steps_to_ori_current_target,
            "current_target_idx": self.current_target_idx,
            "current_pos": self.current_pos,
            "current_ori": self.current_ori,
            "current_target_pos": self.current_target_pos,
            "current_target_ori": self.current_target_ori,
            "is_last_target": self.is_last_target(),
            "visited_points": self.visited_points,
            "visited_orientations": self.visited_orientations
        }
        return {**reset_info, **super()._get_reset_info()}

    
    @property
    def current_pos(self):
        #TODO: change from world frame to base frame
        pos = self.data.site(self.needle_holder_site_id).xpos 
        pos = clean_null_noise(pos, self.null_threshold) * 1e3 # in mm conversion
        pos_in_base_frame = T_global2base @ np.concatenate([pos, [1]])
        return np.round(pos_in_base_frame[:3], 2)
        #return self.data.site(self.needle_holder_site_id).xpos * 1e3 # in mm conversion
    
    @property
    def current_ori(self):

        #TODO: change from world frame to base frame
        ori = self.data.site(self.needle_holder_site_id).xmat.reshape(3, 3)
        ori = clean_null_noise(ori, self.null_threshold)
        ori_in_base_frame = T_global2base[:3,:3] @ ori
        return np.round(ori_in_base_frame, 3)
        #return self.data.site(self.needle_holder_site_id).xmat.reshape(3, 3) 

    @property
    def pos_err_vec(self):
        return np.round(np.array(self.current_target_pos) - np.array(self.current_pos), 2)
    def get_pos_err_vec_at(self, index):
        return np.round(np.array(self.get_target_pos_at(index)) - np.array(self.current_pos), 2)
    
    @property
    def pos_err(self):
        # compute the position error as the Euclidean distance between the current position and the target position
        return np.linalg.norm(self.pos_err_vec)
    
    def get_pos_err_at(self, index):
        return np.linalg.norm(self.get_pos_err_vec_at(index))
   
    @property
    def ori_err_mat(self):
        mat = np.dot(self.current_ori.T, self.current_target_ori)
        return np.round(clean_null_noise(mat, self.null_threshold), 3)
    
    def get_ori_err_mat_at(self, index):
        mat = np.dot(self.current_ori.T, self.get_target_ori_at(index))
        return np.round(clean_null_noise(mat, self.null_threshold), 3)

    @property
    def ori_err(self):
        # compute the orientation error as the angle between the current orientation and the target orientation in radians
        trace_val = np.sum(self.current_ori * self.current_target_ori) # trace with Frobenius inner product
        cos_theta = np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0)
        #assert np.arccos(cos_theta) == np.arccos(np.clip((np.trace(self.ori_err_mat) - 1.0) / 2.0, -1.0, 1.0)), "Both Frobenius inner product and np.trace() must produce the same result. check the values and comment the assert if the error is due to rounding errors"        

        return np.round(np.arccos(cos_theta), 4) # in radians
    
    # determine what should be added to the observation
    # for example, the velocities and positions of various joints can be obtained through their names, as stated here
    def _get_obs(self):
        
        #self.current_pos = self.data.site(self.needle_holder_site_id).xpos*1e3 #convert to mm for consistency with the trajectory points
        #self.current_ori = self.data.site(self.needle_holder_site_id).xmat.reshape(3, 3)
        #self.pos_err_vec = self._get_pos_err()#np.array(self.current_target_pos) - np.array(self.current_pos)

        #self.ori_err_mat = self._get_ori_err_mat()#self.dot(self.current_ori.T, self.current_target_ori)
        obs = np.concatenate([
                            # cartesian postions error
                      
                            self.pos_err_vec,
                            # orientation of the end effector
                            np.array(self.ori_err_mat[:,:2].T.flatten()), # 6D representation of 3 x 3 orientation (2 first column). 
                                                                          #the last column must be the result of the cross product of 1st column(tangent) with second(binormal)
                            #TODO: Add a look ahead mechanism
                            self.get_lookahead_target().flatten(),
                            # joint positions, velocities and control signals
                            np.array(self.data.qpos),
                            np.array(self.data.qvel),

                            ],
                            axis=0)
        return obs
    
    def get_lookahead_target(self,num=3, lookahead_offset=1):
        """
        Récupère un point futur avec saturation à la fin de la liste.
        """
        max_idx = self.num_points - 1
        look_ahead = []
        for i in range(1, num + 1):
            # Calcul de l'index désiré
            desired_idx = self.current_target_idx + i * lookahead_offset
            
            # Saturation (clamping) à l'index maximum
            safe_idx = min(desired_idx, max_idx)
            
            look_ahead.append(np.array(self.get_pos_err_vec_at(safe_idx)))
            look_ahead.append(np.array(self.get_ori_err_mat_at(safe_idx)[:,:2].T.flatten()))


        #print(look_ahead)
        return np.concatenate(look_ahead)
    def close(self):
        # 1. Close MuJoCo / GLFW viewer if it exists
        if hasattr(self, "mujoco_renderer") and self.mujoco_renderer is not None:
            self.mujoco_renderer.close()
            self.mujoco_renderer = None

        if hasattr(self, "viewer") and self.viewer is not None:
            self.viewer.close()
            self.viewer = None

        # 2. Call superclass close
        super().close()
# Exemple d'utilisation dans la boucle step :
# target_pos_i10 = get_lookahead_target(self.traj_points, self.current_target_idx, 10)
if __name__ == "__main__":
    # gym.register(id="my_master_envs/ArmRobot-v1", entry_point="trajectory_env:ArmRobot")
    # env = gym.make("my_master_envs/ArmRobot-v1")
    # env.reset()
    env = ArmRobot(render_mode="human")
    env.reset()
    try:
        #check_env(env, warn=True, skip_render_check=True) 
        init_joints_degrees = np.array([-21.58, -97.97, -66.38, -106.18, 88.76, -21.74])#[0]*6#[-10.042, -93.752, -63.163, -109.568, 76.181, -11.005]
        init_joints_grad = [angle * np.pi / 180 for angle in init_joints_degrees]
        env.data.ctrl[:] = init_joints_grad  # Reset control signals to zero
 # Check if the environment follows Gymnasium's API
        env.data.qpos[:] = init_joints_grad  # Reset control signals to zero
        mujoco.mj_forward(env.model, env.data)  # Forward the simulation to update positions and velocities
        print("ctrl: ",env.data.ctrl.copy())
        print("qpos: ",env.data.qpos.copy())
        print("xpos: ",env.data.site_xpos.copy())
        print("xpos 2: ", env.current_pos)
    except Exception as e:
        print(f"Environment check failed: {e}")
    finally:
        env.close()
    # obs, info = env.reset()
    # for _ in range(1000):
    #     action = env.action_space.sample()  # Sample random action
    #     obs, reward, done, truncated, info = env.step(action)
    #     env.render()
    #     if done or truncated:
    #         obs, info = env.reset()
    # env.close()