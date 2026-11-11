# Reinforcement Learning Control for a UR3e Arm

This project explores reinforcement-learning-based trajectory tracking for a simulated Universal Robots UR3e arm. The robot is simulated with MuJoCo, and the custom Gymnasium environment controls the needle-holder end-effector as it follows a sampled elliptical path while matching a desired orientation.

> **Scope:** This repository contains simulation and research code. The precision values below are environment settings or project goals, not claims about measured or real-robot performance.

## Results

Replace the placeholders with figures and recordings generated from your experiments. Add the media files at the paths shown (or update the paths here).

### Simulation image

![Placeholder: add a screenshot of the UR3e tracking the target trajectory](docs/images/trajectory-tracking.png)

*Suggested figure: show the UR3e, target path, end-effector path, and target checkpoints.*

### Simulation video

<!-- Add a video at docs/videos/trajectory-tracking.mp4, or change the source below. -->
<video controls width="720" src="docs/videos/trajectory-tracking.mp4">
  Your browser cannot play this video. Add a link to the simulation recording here.
</video>

*Suggested recording: show one complete evaluation episode and include the policy/checkpoint name.*

## Project overview

- **Robot model:** UR3e MJCF model and scene in `universal_robots_ur3e-main/`.
- **Environment:** `theEnv31.py` implements the `ArmRobot` Gymnasium environment.
- **Trajectory and frames:** `trajectory.py` computes the elliptical target path and coordinate-frame transforms.
- **Agent manager:** `theAgent31.py` provides Stable-Baselines3 Soft Actor-Critic (SAC) training and evaluation helpers.
- **Control:** the agent emits six normalized joint increments. The environment scales them to joint-angle changes and applies the resulting controls to MuJoCo.
- **Observation:** a 34-value state containing joint positions and velocities, end-effector position error and path information, and orientation-error representations.

The environment uses a default episode limit of 200 steps, a 20-point trajectory, and a 50-micrometer position-reach threshold. These are configurable; see the `ArmRobot` constructor in `theEnv31.py`. The trajectory dimensions and frame transformations are defined in `trajectory.py`.

## Requirements

- Python
- MuJoCo 2.3.3 or later (the bundled robot model's stated requirement)
- Gymnasium with its MuJoCo dependencies
- Stable-Baselines3, NumPy, SciPy, Matplotlib, and TensorBoard
- ImageIO for the video/visualization utilities in `test_env.py`

There is currently no populated dependency lockfile or `requirements.txt`. In a virtual environment, install the packages with:

```bash
python -m pip install "gymnasium[mujoco]" stable-baselines3 numpy scipy matplotlib tensorboard imageio
```

If rendering fails in a headless environment, configure a MuJoCo-compatible offscreen rendering backend for your system. Human-window rendering requires a working graphical display.

## Setup

Run commands from the project directory so that the environment can find `universal_robots_ur3e-main/scene.xml` using its relative path.

```bash
cd rl_control_armrobot
```

The repository includes the robot model and several trained model files under `saved_models/`. Model files and training logs can be large; keep your experiment outputs organized and record which checkpoint produced each result.

## Run a quick environment check

This creates the environment directly, resets it, takes one random action, and closes it:

```bash
python - <<'PY'
from theEnv31 import ArmRobot

env = ArmRobot()
try:
    observation, info = env.reset()
    print("Observation shape:", observation.shape)
    observation, reward, terminated, truncated, info = env.step(
        env.action_space.sample()
    )
    print("Reward:", reward)
finally:
    env.close()
PY
```

## Evaluate a trained policy

From the project directory, run:

```bash
python theAgent31.py
```

The script registers the `ArmRobot` environment, loads the SAC model named `nolook_envTv36` from `saved_models/`, and evaluates it for five episodes. To evaluate another checkpoint, change the `name_p` value in the `SACAgentManager` initialization near the bottom of `theAgent31.py`. The checkpoint must be present in `saved_models/` and match the environment and policy configuration.

In `test_env2.py`, each completed episode also saves its per-step cross-track and orientation errors under `tracking_error_results/`. `tracking_error_samples.csv` contains the sample values; `tracking_error_metrics.csv` contains per-episode MSE and RMSE (cross-track error in mm, orientation error in degrees). Orientation error is calculated from the environment's interpolated-path `cos_theta` value.

## Train a policy

In the `if __name__ == "__main__":` section of `theAgent31.py`:

1. Change `loadmodel=True` to `loadmodel=False` when creating `SACAgentManager` to start a new policy.
2. Enable the `manager.train(...)` call and choose the desired `total_timesteps` and TensorBoard run name.
3. Keep the evaluation call optional; it uses a separate environment instance.

Training saves the resulting SAC policy under `saved_models/`, checkpoints under `sac_eval_checkpoints/`, and TensorBoard logs under `tb_logs/`.

## Configuration and notes

`ArmRobot` accepts parameters for the trajectory, workspace, thresholds, initial target, episode length, rendering, and initial joint-position noise. For example, instantiate it with `ArmRobot(render_mode="human")` to open a MuJoCo viewer when a display is available.

The standalone script `test_env.py` contains visualization and experiment code in addition to environment setup. Prefer the quick check above for a minimal smoke test; inspect the script before running it because it executes simulation code at module level.

## Read the real robot TCP pose

To read the current tool-center-point (TCP) position and orientation from a UR robot, run the separate read-only script:

```bash
python get_robot_pose.py --host ROBOT_IP
```

Replace `ROBOT_IP` with the robot controller's IP address. The script prints position in meters and millimeters, the orientation rotation vector in radians and degrees, a rotation matrix, and XYZ Euler angles in degrees. The pose is reported relative to the robot base and uses the TCP configured on the controller. Ensure the configured TCP matches the tool you want to measure.

## Project goals

The project requirements include tracking an elliptical path, reducing position error/RMSE, smooth control, and investigating sim-to-real transfer and collision-free behavior. These are research goals; successful simulation does not establish hardware safety or sim-to-real performance.
