from gymnasium.envs import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import os
from trajectory import elliptical_trajectory
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
    def __init__(self, path_start_idx=0, max_joint_delta = 1.0, episode_len=500, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        # change shape of observation to your observation space size
        observation_space = Box(low=-np.inf, high=np.inf, shape=(30,), dtype=np.float64)

        # world space, reduced3D space representing a reduction of the workspace of the robot
        self.world_space = Box(low=np.array([-1.0, -1.0, 0.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float64)

        # load your MJCF model with env and choose frames count between actions
        MujocoEnv.__init__(
            self,
            os.path.abspath("universal_robots_ur3e-main/scene.xml"),
            5,
            observation_space=observation_space,
            **kwargs
        )

        bounds = self.model.actuator_ctrlrange.copy().astype(np.float64)
        self.actions_low, self.actions_high = bounds.T
        num_actions = len(self.actions_low)
        self._max_joint_delta = max_joint_delta # in degrees, maximum change in joint angles per step
        
        # Override the action space for the RL agent
        delta_bounds = np.ones(num_actions, dtype=np.float64) * self._max_joint_delta * np.pi / 180  # Convert degrees to radians
        self.action_space = Box(
            low=-delta_bounds, 
            high=delta_bounds, 
            dtype=np.float64
        )

        self.step_number = 0
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
        self.current_pos = None # np.array([self.target_path[0,0], self.target_path[1,0], self.target_path[2,0]])
        self.current_ori = None # np.array([self.target_orientations[0,0], self.target_orientations[1,0], self.target_orientations[2,0]])

        self.visited_points = [False] * len(self.target_path)
        #self.visited_points[0] = True

        self.target_pos_index = self.path_start_idx
        self.target_ori_index = self.path_start_idx
        
    @property
    def current_target_pos(self):
        return np.array(self.target_path[:,self.target_pos_index])
    
    @property
    def current_target_ori(self):
        return  np.array([self.target_tangent[:,self.target_ori_index],
                         self.target_binormal[:,self.target_ori_index],
                         self.target_normal[:,self.target_ori_index]]).T
    #def get_p
    # determine the reward depending on observation or other properties of the simulation
    def step(self, d_action):
        #reward = 1.0

        current_ctrl = self.data.ctrl.copy()
        
        # B. Apply delta and clip to XML ctrlrange boundaries
        absolute_action = current_ctrl + d_action
        clipped_action = np.clip(
            absolute_action, 
            self.actions_low, 
            self.actions_high
        )

        # perform the simulation step with the given action and frame_skip 
        self.do_simulation(clipped_action, self.frame_skip)
        self.step_number += 1

        # get the observation after the simulation step
        obs = self._get_obs()

        # compute the reward and the termination conditions
        reward, terminated = self._get_reward(obs)

        truncated = self.step_number > self.episode_len
        return obs, reward, terminated, truncated, {}
        
    def _get_reward(self, obs):
        #TODO: complete the reward function to compute the reward based on the observation and other properties of the simulation

        terminated =  False
        
        reward = 0#... # compute the reward based on the observation and other properties of the simulation

        # check termination conditions, 
        # for example if the end effector is too far from the target or if the episode length is reached
        # or if the robot reaches the target position and orientation, then update the target to the next point in the trajectory
        terminated = True #... # check termination conditions
        return reward, terminated

    # define what should happen when the model is reset (at the beginning of each episode)
    def reset_model(self):
        # TODO: complete the reset_model function to reset the robot to its initial state and set the target to the first point in the trajectory
        self.step_number = 0
        self.current_target_ori_index = self.path_start_idx
        self.current_target_pos_index = self.path_start_idx
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
        return super()._get_reset_info()
    
    # determine what should be added to the observation
    # for example, the velocities and positions of various joints can be obtained through their names, as stated here
    def _get_obs(self):
        site_id = self.model.site("attachment_site").id
        self.current_pos = self.data.site(site_id).xpos
        self.current_ori = self.data.site(site_id).xmat.reshape(3, 3) @ np.array([0, 1, 0])
        obs = np.concatenate([
                            # cartesian postions
                            np.array(self.current_pos),
                            np.array(self.current_target_pos),
                            # oreintation of the end effector
                            np.array(self.current_ori),
                            np.array(self.current_target_ori),
                            # joint positions, velocities and control signals
                            np.array(self.data.qpos),
                            np.array(self.data.qvel),
                            np.array(self.data.ctrl) ,

                            ],
                            axis=0)
        return obs
    
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