import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.utils import EzPickle

class NewStyleRobotEnv(MujocoEnv, EzPickle):
    # 1. Define standard Gymnasium rendering formats
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 50,
    }

    def __init__(self, xml_file="robot_scene.xml", frame_skip=5, **kwargs):
        # 2. Correct constructor handling using EzPickle for serialization
        EzPickle.__init__(self, xml_file, frame_skip, **kwargs)
        
        # 3. Pass full path or package-relative assets directly to MujocoEnv
        # In v5+, default_camera_config can be customized via a dict
        super().__init__(
            model_path=xml_file,
            frame_skip=frame_skip,
            observation_space=None,  # Automatically inferred if generated below
            default_camera_config={"distance": 3.0, "elevation": -20.0},
            **kwargs
        )

    def step(self, action):
        # 4. Step the simulation forward using frame skipping
        self.do_simulation(action, self.frame_skip)
        
        obs = self._get_obs()
        
        # 5. Handle standard 5-tuple returns (Separated Terminal vs Truncated)
        reward = self._compute_reward(action)
        terminated = self._check_termination()
        truncated = False
        info = {"x_position": self.data.qpos[0]}
        
        return obs, reward, terminated, truncated, info

    def reset_model(self):
        # 6. Replaces old model reset pipelines; defines state randomization
        noise_low, noise_high = -0.01, 0.01
        qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv)
        
        # Directly manipulate underlying C structures
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        # 7. Fast, non-copy pointer slicing directly from dynamic simulation memory
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])

    def _compute_reward(self, action):
        return 1.0  # Custom reward function logic here
        
    def _check_termination(self):
        return bool(self.data.qpos[2] < 0.2)  # Example: Terminate if the robot falls

#%%