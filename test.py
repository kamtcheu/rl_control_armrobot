import time

import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path('./universal_robots_ur3e-main/scene.xml')
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running():# and time.time() - start < 30:
    step_start = time.time()

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d)

    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)

  #%%
  import numpy as np


def dh_transform(a, alpha, d, theta):
    """Standard DH homogeneous transformation matrix."""
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,        sa,       ca,      d],
        [0,         0,        0,      1]
    ])


def fk_ur3e(q):
    """
    Forward kinematics for UR3e using standard DH parameters.

    Parameters
    ----------
    q : array-like, shape (6,)
        Joint angles [q1, q2, q3, q4, q5, q6] in radians

    Returns
    -------
    T : (4,4) ndarray
        Homogeneous transform base -> tool flange
    """

    q = np.asarray(q)

    # UR3e standard DH parameters
    d = np.array([
        0.15185,
        0.0,
        0.0,
        0.13105,
        0.08535,
        0.09210
    ])

    a = np.array([
        0.0,
        -0.24355,
        -0.21320,
        0.0,
        0.0,
        0.0
    ])

    alpha = np.array([
        np.pi / 2,
        0.0,
        0.0,
        np.pi / 2,
        -np.pi / 2,
        0.0
    ])

    T = np.eye(4)

    for i in range(6):
        A_i = dh_transform(a[i], alpha[i], d[i], q[i])
        T = T @ A_i

    return T


# Example usage
if __name__ == "__main__":
    q = np.array([0, -1.51, 1.51, -1.51, -1.51, 0])

    T = fk_ur3e(q)

    print("T_base_tcp =")
    print(np.round(T, 6))

    position = T[:3, 3]
    rotation = T[:3, :3]

    print("\nPosition:")
    print(position)

    print("\nRotation matrix:")
    print(rotation)
  
# %%
