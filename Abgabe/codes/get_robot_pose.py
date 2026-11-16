import argparse

import numpy as np
from scipy.spatial.transform import Rotation
import urx


def main():
    parser = argparse.ArgumentParser(
        description="Read the current TCP position and orientation from a UR robot."
    )
    parser.add_argument(
        "--host",
        required=True,
        help="IP address or hostname of the robot controller.",
    )
    args = parser.parse_args()

    robot = urx.Robot(args.host)
    try:
        pose_info = robot.secmon.get_cartesian_info()#np.asarray(, dtype=np.float64)
        print(f"Robot returned Cartesian pose: {pose_info}")
        # if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        #     raise RuntimeError(f"Robot returned an invalid Cartesian pose: {pose}")
    finally:
        robot.close()

    position_m = np.array([pose_info["X"], pose_info["Y"], pose_info["Z"]])
    rotation_mat = Rotation.from_euler('xyz', np.array([pose_info["Rx"], pose_info["Ry"], pose_info["Rz"]]))
    #rotation = Rotation.from_rotvec(rotation_vector_rad)

    np.set_printoptions(precision=6, suppress=True)
    print("TCP position in robot base frame:")
    print(f"  meters:      {position_m}")
    print(f"  millimeters: {position_m * 1000}")
    print("TCP orientation in robot base frame:")
    print(f"  rotation matrix:")
    print(rotation_mat.as_matrix())
    print(f"  Euler XYZ (deg): {rotation_mat.as_euler('xyz', degrees=True)}")


if __name__ == "__main__":
    main()
