from gymnasium.envs import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import os
from trajectory import elliptical_trajectory, T_global2base, elipse_center_global
from gymnasium.utils.env_checker import check_env




# you can completely modify this class for your MuJoCo environment by following the directions
class ArmRobot(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    # set default episode_len for truncate episodes 
    # eps_pos and eps_ori are the thresholds for position and orientation errors to consider the target reached
    # eps_pos is in meters and eps_ori is in degrees
    # max_joint_delta represents the max allowed angle adjustement per step and is in in degrees
    def __init__(self,work_box_size =0.1, threshold_pos=60e-6, threshold_ori=1, path_start_idx=0, max_joint_delta = 1.0, render_mode="human", site_name="needle_holder_site", episode_len=500, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        # change shape of observation to your observation space size
        observation_space = Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float64)

        self.ellipse_center_base = T_global2base@(np.concatenate([np.array(elipse_center_global).flatten(),[1]]))
        self.ellipse_center_base = self.ellipse_center_base[:3] # in mm
        half_extents = np.array([work_box_size/2, work_box_size/2, work_box_size/2], dtype=np.float64) * 1e3 # convert in mm

        # work space
        self.work_space = Box(
            low=self.ellipse_center_base - half_extents,
            high=self.ellipse_center_base + half_extents,
            dtype=np.float64
        )



        self.threshold_pos = threshold_pos*1e3 # convert to mm
        self.threshold_ori = threshold_ori*np.pi/180 # convert to radians
        self.xml_needle_holder_name = site_name

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
        delta_bounds = np.ones(num_actions, dtype=np.float64) * (self._max_joint_delta * np.pi / 180)  # Convert degrees to radians
        self.action_space = Box(
            low=-delta_bounds, 
            high=delta_bounds, 
            dtype=np.float64
        )

        self.step_number = 0
        self.steps_to_pos_current_target = 0
        self.steps_to_ori_current_target = 0
        self.episode_len = episode_len
        self.path_start_idx = path_start_idx
        #self.init_qpos = np.zeros(self.model.nq)#self.model.key_qpos[0].copy() # joint positions of the robot at the beginning of the episode(first point of the trajectory)
        
        #Init joints in degrees and convert to radians from autonomous_surgery repo 
        #TODO: fine tune to match the pose for the first point of the trajectory
        init_joints_degrees = [-10.042, -93.752, -63.163, -109.568, 76.181, -11.005]
        init_joints_grad = [angle * np.pi / 180 for angle in init_joints_degrees]
        self.init_qpos = init_joints_grad #self.model.keyframe("home").qpos.copy() # joint positions of the robot at the beginning of the episode(first point of the trajectory)

        # precomputed target trajectory and orientation of the TCP given in the base frame in mm
        _ , trajectory_in_base_frame = elliptical_trajectory(a=5, b=3, N=19, show_plot=True)
        self.target_path, self.target_tangent, self.target_binormal, self.target_normal = trajectory_in_base_frame

        # will be set at reset time to a point near the first point of the trajectory( for the moment,
        # it will be set to the first trajectory point later, but it can be set to a random point near the first trajectory point)
        # self.current_pos = None # np.array([self.target_path[0,0], self.target_path[1,0], self.target_path[2,0]])
        # self.current_ori = None # np.array([self.target_orientations[0,0], self.target_orientations[1,0], self.target_orientations[2,0]])

        # self.p_err = None
        # self.ori_err = None
        self.w_pos = 0.5
        self.w_ori = 0.5
        self.w_cost = 0.01 # weight for the control cost in the reward function
        self.visited_points = [False] * len(self.target_path[0])
        self.visited_orientations = [False] * len(self.target_path[0])
        #self.visited_points[0] = True

        # self.target_pos_index = self.path_start_idx
        # self.target_ori_index = self.path_start_idx
        self.current_target_idx = self.path_start_idx
        
    @property
    def current_target_pos(self):
        # shape 3 x 1 in mm
        return np.array(self.target_path[:,self.current_target_idx])
    
    @property
    def current_target_ori(self):
        # shape 3 x 3
        return  np.array([self.target_tangent[:,self.current_target_idx],
                         self.target_binormal[:,self.current_target_idx],
                         self.target_normal[:,self.current_target_idx]]).T
    #def get_p
    # determine the reward depending on observation or other properties of the simulation
    def step(self, d_action):

        # fetch current control value
        current_ctrl = self.data.ctrl.copy()

        # add the delta. Noise will be handle by SB3 Wrapper during training. It will be added to d_action before passing it to / calling "step()"
        absolute_action = current_ctrl + d_action

        # Apply delta and clip to XML ctrlrange boundaries
        clipped_action = np.clip(
            absolute_action, 
            self.actions_low, 
            self.actions_high
        )

        # perform the simulation step with the given action and frame_skip 
        self.do_simulation(clipped_action, self.frame_skip)

        # step counter
        self.step_number += 1

        # update the number of steps taken to reach the current target position and orientation
        # until the target is reached, if the target is reached, the counter will not be updated anymore until the next target is set
        self.steps_to_pos_current_target += 1*(not self.visited_points[self.current_target_idx])
        self.steps_to_ori_current_target += 1*(not self.visited_orientations[self.current_target_idx])

        # get the observation after the simulation step
        obs = self._get_obs()

        # compute the reward and the termination conditions
        reward, rew_info = self._get_reward(d_action)

        self.update_visited_checkpoints()
        self.update_pos_ori_weights()
        self.update_current_target()
        terminated = self.check_termination_conditions()
        truncated = self.step_number > self.episode_len

        info = {
            "step_number": self.step_number,
            "steps_to_pos_current_target": self.steps_to_pos_current_target,
            "steps_to_ori_current_target": self.steps_to_ori_current_target,
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

        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info


    def _get_crtl_cost(self, d_action):
        # compute the control cost based on the action taken
        # for example, the control cost can be computed as the L2 norm of the action
        ctrl_cost = np.sum(np.square(d_action))
        return ctrl_cost
    
    def _get_position_reward(self):
        sigma_p = self.threshold_pos
        r_pos = 0.5 * np.exp(- (self.pos_err ** 2) / ((10*sigma_p)** 2)) + 0.5 * np.exp(- (self.pos_err ** 2) / ( sigma_p** 2)) # Gaussain RBF, radial basis function
        return r_pos
    
    def _get_orientation_reward(self):
        # o_err_mat = self._get_ori_err_mat()
        # trace_val = np.sum(self.current_ori * self.current_target_ori) # trace with Frobenius inner product
        # cos_theta = np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0)
        # self.ori_err = np.arccos(cos_theta)
        # assert self.ori_err == np.arccos(np.clip((np.trace(self.ori_err_mat) - 1.0) / 2.0, -1.0, 1.0)), "Both Frobenius inner product and np.trace() must produce the same result. check the values and comment the assert if the error is due to rounding errors"
        r_ori = (np.trace(self.ori_err_mat) + 1) / 4
        return r_ori

    def update_visited_checkpoints(self):
        if self.ori_err < self.threshold_ori:
            self.visited_orientations[self.current_target_idx] = True
        if self.pos_err < self.threshold_pos:
            self.visited_points[self.current_target_idx] = True

    def update_pos_ori_weights(self):
        if self.visited_points[self.current_target_idx] ^ self.visited_orientations[self.current_target_idx]:
            self.w_pos, self.w_ori = 0.3, 0.7 if self.visited_points[self.current_target_idx] else 0.7, 0.3
        else:
            self.w_pos, self.w_ori = 0.5, 0.5

    def current_target_reached(self):
        return self.visited_points[self.current_target_idx] and self.visited_orientations[self.current_target_idx]

    @property
    def bonus_current_target_reached(self):
        # return a bonus reward if the current target is reached, with a higher bonus if both position and orientation are reached at the same time step
        bonus = 0.0
        if self.current_target_reached():
            bonus = 1.0 if self.steps_to_ori_current_target == self.steps_to_pos_current_target else 0.5 # bonus for reaching the target
            if self.is_last_target():
                bonus += 1.0 # additional bonus for reaching the last target
        return bonus
    
    @property
    def penalty_leaving_workspace(self):
        # return a penalty if the robot leaves the workspace
        penalty = 0.0
        if not self.work_space.contains(self.current_pos):
            penalty = -1.0 #if self.steps_to_pos_current_target > 10 else -0.5 # penalty for leaving the workspace
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
            if self.current_target_idx < len(self.target_path[0]) - 1:
                self.current_target_idx += 1
                self.w_pos, self.w_ori = 0.5, 0.5
                self.steps_to_pos_current_target = 0
                self.steps_to_ori_current_target = 0
            # else:
            #     # if the last point of the trajectory is reached, the episode can be terminated
            #     pass

    # def last_target_reached(self):
    #     return self.current_target_idx == (len(self.target_path[0]) - 1) and self.current_target_reached()
    
    def is_last_target(self):
        return self.current_target_idx == (len(self.target_path[0]) - 1)
    
    def check_termination_conditions(self):
        # check termination conditions, 
        # for example if the end effector is too far from the target or if the episode length is reached
        # or if the robot reaches the target position and orientation, then update the target to the next point in the trajectory
        terminated = False
        if not self.work_space.contains(self.current_pos):
            terminated = True
        elif self.step_number > self.episode_len:
            terminated = True
        elif self.is_last_target() and self.current_target_reached(): #last target reached
            terminated = True
        #TODO: Add a termination condition if the robot is too far from the target position or orientation for too long
        return terminated  

    def _get_reward(self, d_action):

        # R POSITION compute position error reward
        r_pos = self._get_position_reward()#obs[:3])
        # R ORIENTATION compute orientation reward
        r_ori = self._get_orientation_reward()
        # cost JOINT VELOCITY PENALTY
        c_vel = -self._get_crtl_cost(d_action)

        bonus = self.bonus_current_target_reached
        penalty = self.penalty_leaving_workspace


        reward = self.w_pos*r_pos + self.w_ori*r_ori + self.w_cost*c_vel + bonus + penalty  #... # compute the reward based on the observation and other properties of the simulation
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

        self.steps_to_pos_current_target = 0
        self.steps_to_ori_current_target = 0

        self.current_target_idx = self.path_start_idx
        # self.target_ori_index = self.path_start_idx
        # self.current_pos = np.array([self.target_path[0,0], self.target_path[1,0], self.target_path[2,0]])
        # self.current_ori = np.array([self.target_tangent[0,0], self.target_tangent[1,0], self.target_tangent[2,0]])

        # self.model.keyframe('keyframe_name').id
        # Clears the slate and loads the XML's keyframe 0 configuration
        # mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        
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
            "steps_to_pos_current_target": self.steps_to_pos_current_target,
            "steps_to_ori_current_target": self.steps_to_ori_current_target,
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
        pos = self.data.site(self.needle_holder_site_id).xpos * 1e3 # in mm conversion
        base_pos = T_global2base @ np.concatenate([pos, [1]])
        return base_pos[:3]
        #return self.data.site(self.needle_holder_site_id).xpos * 1e3 # in mm conversion

    @property
    def current_ori(self):
        #TODO: change from world frame to base frame
        ori = self.data.site(self.needle_holder_site_id).xmat.reshape(3, 3)
        base_ori = T_global2base[:3,:3] @ ori
        return base_ori
        #return self.data.site(self.needle_holder_site_id).xmat.reshape(3, 3) 

    @property
    def pos_err_vec(self):
        return np.array(self.current_target_pos) - np.array(self.current_pos)
    
    @property
    def pos_err(self):
        # compute the position error as the Euclidean distance between the current position and the target position
        return np.linalg.norm(self.pos_err_vec)
   
    @property
    def ori_err_mat(self):
        return np.dot(self.current_ori.T, self.current_target_ori)
    
    @property
    def ori_err(self):
        # compute the orientation error as the angle between the current orientation and the target orientation in radians
        trace_val = np.sum(self.current_ori * self.current_target_ori) # trace with Frobenius inner product
        cos_theta = np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0)
        assert np.arccos(cos_theta) == np.arccos(np.clip((np.trace(self.ori_err_mat) - 1.0) / 2.0, -1.0, 1.0)), "Both Frobenius inner product and np.trace() must produce the same result. check the values and comment the assert if the error is due to rounding errors"        

        return np.arccos(cos_theta)
    
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

                            # joint positions, velocities and control signals
                            np.array(self.data.qpos),
                            np.array(self.data.qvel),

                            ],
                            axis=0)
        return obs
    
    def get_lookahead_target(self, lookahead_offset=3):
        """
        Récupère un point futur avec saturation à la fin de la liste.
        """
        max_idx = len(self.target_path[0]) - 1
        
        # Calcul de l'index désiré
        desired_idx = self.current_target_idx + lookahead_offset
        
        # Saturation (clamping) à l'index maximum
        safe_idx = min(desired_idx, max_idx)
        
        return self.target_path[safe_idx]

# Exemple d'utilisation dans la boucle step :
# target_pos_i10 = get_lookahead_target(self.traj_points, self.current_target_idx, 10)
if __name__ == "__main__":
    env = ArmRobot()
    check_env(env, warn=True)  # Check if the environment follows Gymnasium's API
    # obs, info = env.reset()
    # for _ in range(1000):
    #     action = env.action_space.sample()  # Sample random action
    #     obs, reward, done, truncated, info = env.step(action)
    #     env.render()
    #     if done or truncated:
    #         obs, info = env.reset()
    # env.close()