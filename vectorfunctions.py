import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as linalg
from copy import deepcopy
import math3d as m3d

#import suture_process
import urx
import ur3
from math import sin, cos, sqrt, radians


# Init all variable and parameters


# The homogeneous transformation matrices between the Global Frame and the two other Body Frames
#     for Robot1 and Robot2
#     Robot1ThMatrix and Robot2ThMatrix allow the transformation from the Global Frame to the
#     Robotx Frames.
#     Robot1ThMatrixInv and Robot2ThMatrixInv allow the transformation from the Robotx Frames to
#     the Global Frames
#     All these variables are declared "global" and may be used overall
#     WARNING: In case the robots are repositioned or the origin of the Global Frame are moved or rotated
#              then the data needs to be updated

rob1_ip = "141.64.75.55"
rob2_ip = "141.64.75.56"

a_joints = 0.2
v_joints = 0.2

global_frame_origin = np.zeros([3, 1])


# Definition of all functions of this library


def mirror_joints1to2(joint_1):
    """Function to mirror the joint angles of Robot1 so that Robot2 seems to be mirrored
        on a vertical plane between the two robots"""
    factory_home_position_joints = np.array([0.0, -90.0, 0.0, -90.0, 0.0, 0.0]) * np.pi / 180
    delta = factory_home_position_joints - joint_1
    joint_2 = joint_1 + 2 * delta

    return joint_2


def compute_orientation_matrix(angle_x, angle_y):
    """Function to compute the rotation matrix given the two angles AngleX and AngleY; both
        angles are in radians.
        IMPORTANT_NOTE 1: The sum of both angles MUST be larger than pi / 2, i.e. 90 degrees
        IMPORTANT_NOTE 2: The x and y-axis are referred to the Global Frame, i.e. the table
        Inputs:
                AngleX: AngleX is the angle between the desired orientation and the x-axis
                AngleY: AngleY is the angle between the desired orientation and the y-axis
        Outputs:
                OrientationMatrix: It is the rotation matrix that represents the orientation of the gripper.
    """
    axis_deg = np.array([90, 135, 45])
    axis_0 = np.cos(axis_deg * np.pi / 180)
    angle_0 = np.pi
    orientation_matrix_0 = compute_rotation_matrix(axis_0, angle_0)
    a_vector = np.array([0.0, 0.0, 1.0])
    b_vector = np.array([0.0, 0.0, 0.0])
    b_vector[0], b_vector[1] = np.cos(angle_y), np.cos(angle_x)
    b_vector[2] = np.sqrt(1.0 - np.square(b_vector[0]) - np.square(b_vector[1]))
    rot_axis = np.cross(a_vector, b_vector)
    rot_axis = rot_axis / linalg.norm(rot_axis)
    rot_angle = np.arccos(np.dot(a_vector, b_vector))
    orientation_matrix = compute_rotation_matrix(rot_axis, rot_angle)
    orientation_matrix = np.matmul(orientation_matrix_0, orientation_matrix)

    return orientation_matrix


def translate_trajectory(trajectory_in, new_coordinate_sys):
    """Function to translate a trajectory (Trajectory) from one coordinate system to one another
        The trajectory to be translated given as a numpy array 3 x (length of trajectory)
        The new coordinate system is given by a 3D vector 3 x 1
        The output (NewTrajectory) has the same dimensions of Trajectory"""

    dimensions = trajectory_in.shape
    trajectory_out = np.zeros(dimensions)
    for i in range(dimensions[1]):
        trajectory_out[:, i:(i+1)] = trajectory_in[:, i:(i + 1)] + new_coordinate_sys

    return trajectory_out


def scale_trajectory(trajectory_in, scale):
    """Function to scale a trajectory. The scale factors for each dimension may be different
        The first point of the trajectory remains in the same point and is used as reference
        Input:
                Trajectory_in is a matrix (3xX) that contains the point of the trajectory
                Scale is an array (1x3) that contains the scaling factors for the three dimensions
                x, y, z.
        Output:
                Trajectory_out is a matrix (3xX) that contains the scaled trajectory"""

    dimensions = trajectory_in.shape
    x_scale, y_scale, z_scale = scale[0], scale[1], scale[2]
    x_ref, y_ref, z_ref = trajectory_in[0, 0], trajectory_in[1, 0], trajectory_in[2, 0]
    trajectory_out = np.zeros(dimensions)
    trajectory_out[0, :] = x_scale * (trajectory_in[0, :] - x_ref) + x_ref
    trajectory_out[1, :] = y_scale * (trajectory_in[1, :] - y_ref) + y_ref
    trajectory_out[2, :] = z_scale * (trajectory_in[2, :] - z_ref) + z_ref

    return trajectory_out


def transform_from_global_to_robot1(global_coordinates):
    """Function to transform a trajectory in Global coordinates to Robot1 coordinates
      Input: A position trajectory referred to the Global frame (3xX)
      Output: A position trajectory referred to the Robot1 frame (3xX)"""

    dimensions = global_coordinates.shape
    robot_coordinates = np.zeros(dimensions)
    robot1_th_matrix = create_transformation_matrices()[0]
    for i in range(dimensions[1]):
        g_coordinates = global_coordinates[:, i:(i + 1)]
        g_coordinates = np.append(g_coordinates, np.array([1]).reshape(1, 1), axis=0)
        r_coordinates = np.matmul(robot1_th_matrix, g_coordinates)
        r_coordinates = np.delete(r_coordinates, 3, 0)
        robot_coordinates[:, i:(i+1)] = r_coordinates

    return robot_coordinates


def transform_from_global_to_robot2(global_coordinates):
    """Function to transform a trajectory in Global coordinates to Robot2 coordinates
      Input: A position trajectory referred to the Global frame (3xX)
      Output: A position trajectory referred to the Robot1 frame (3xX)"""
    robot2_th_matrix = create_transformation_matrices()[1]
    dimensions = global_coordinates.shape
    robot_coordinates = np.zeros(dimensions)
    for i in range(dimensions[1]):
        g_coordinates = global_coordinates[:, i:(i + 1)]
        g_coordinates = np.append(g_coordinates, np.array([1]).reshape(1, 1), axis=0)
        r_coordinates = np.matmul(robot2_th_matrix, g_coordinates)
        r_coordinates = np.delete(r_coordinates, 3, 0)
        robot_coordinates[:, i:(i+1)] = r_coordinates

    return robot_coordinates


def transform_from_robot1_to_global(robot_coordinates):
    """Function to transform a trajectory in Robot1 coordinates to Global coordinates
      Input: A position trajectory referred to the Robot1 frame (3xX)
      Output: A position trajectory referred to the Global frame (3xX)"""

    dimensions = robot_coordinates.shape
    global_coordinates = np.zeros(dimensions)
    robot1_th_matrix_inv = create_inverse_transformation_matrices()[0]
    for i in range(dimensions[1]):
        r_coordinates = robot_coordinates[:, i:(i + 1)]
        r_coordinates = np.append(r_coordinates, np.array([1]).reshape(1, 1), axis=0)
        g_coordinates = np.matmul(robot1_th_matrix_inv, r_coordinates)
        g_coordinates = np.delete(g_coordinates, 3, 0)
        global_coordinates[:, i:(i+1)] = g_coordinates

    return global_coordinates


def transform_from_robot2_to_global(robot_coordinates):
    """Function to transform a trajectory in Robot2 coordinates to Global coordinates
      Input: A position trajectory referred to the Robot2 frame (3xX)
      Output: A position trajectory referred to the Global frame (3xX)"""

    dimensions = robot_coordinates.shape
    global_coordinates = np.zeros(dimensions)
    robot2_th_matrix_inv = create_inverse_transformation_matrices()[1]
    for i in range(dimensions[1]):
        r_coordinates = robot_coordinates[:, i:(i + 1)]
        r_coordinates = np.append(r_coordinates, np.array([1]).reshape(1, 1), axis=0)
        g_coordinates = np.matmul(robot2_th_matrix_inv, r_coordinates)
        g_coordinates = np.delete(g_coordinates, 3, 0)
        global_coordinates[:, i:(i+1)] = g_coordinates

    return global_coordinates


def mirror_trajectory(trajectory_in, plane):
    """Function to mirror a trajectory using the plane E as mirror
      Input: Trajectory_in is the trajectory to be mirrored (3xX)
             The plane E is defined by one point Pe (3x1)
             and the normal vector to the plane Ne (3x1)
             Pe and Ne are in the array Plane = [Pe' Ne] (3x2)
      Output: Trajectory_out is the mirrored trajectory (3xX)"""

    dimensions = trajectory_in.shape
    pe, ne = plane[:, 0:1], plane[:, 1:2]
    ne = ne / linalg.norm(ne)
    ke_1 = np.identity(3) * ne
    ke_2 = (np.ones([3, 3]) * ne).transpose()
    ke = np.matmul(ke_1, ke_2)
    m1 = 2 * np.matmul(ke, pe)
    m2 = np.identity(3) - 2 * ke

    trajectory_out = np.zeros(dimensions)
    for i in range(dimensions[1]):
        trajectory_out[:, i:(i+1)] = m1 + np.matmul(m2, trajectory_in[:, i:(i + 1)])

    return trajectory_out


def compute_rotation_axis_angle(rotation_matrix):
    """Compute the rotation axis and angle given a 3x3 rotation matrix
      Input: Rotation Matrix is a 3x3 proven rotation matrix
      Output: Rotation angle in rad
              Rotation axis is a unity vector"""

    rotation_angle = np.arccos(0.5 * (np.trace(rotation_matrix) - 1.0))

    rotation_axis = np.zeros([3, 1])
    rotation_axis[0] = rotation_matrix[2, 1] - rotation_matrix[1, 2]
    rotation_axis[1] = rotation_matrix[0, 2] - rotation_matrix[2, 0]
    rotation_axis[2] = rotation_matrix[1, 0] - rotation_matrix[0, 1]
    rotation_axis = rotation_axis / (2 * np.sin(rotation_angle))
    rotation_axis = rotation_axis / linalg.norm(rotation_axis)

    return rotation_axis, rotation_angle


def compute_rotation_matrix(rotation_vector, rotation_angle):
    """Compute the rotation matrix
      Input: Rotation vector is a 3x1 vector
             Rotation angle is a scalar value in rad
      Output: Rotation matrix"""

    r_vector = rotation_vector / linalg.norm(rotation_vector)
    v_matrix = np.zeros([3, 3])
    v_matrix[1, 0] = + r_vector[2]
    v_matrix[2, 0] = - r_vector[1]
    v_matrix[0, 1] = - r_vector[2]
    v_matrix[2, 1] = + r_vector[0]
    v_matrix[0, 2] = + r_vector[1]
    v_matrix[1, 2] = - r_vector[0]
    rotation_matrix = np.identity(3) + np.sin(rotation_angle) * v_matrix + (1.0 - np.cos(rotation_angle)) * np.matmul(
        v_matrix, v_matrix)

    return rotation_matrix


def rotate_trajectory(trajectory_in, line, angle):
    """Function to rotate a trajectory (= Trajectory) around a line (=Line)
      by an angle (=Angle)
      The trajectory to be rotated is given as a 3D vector array 3 x length of the trajectory
      The Line is specified by means of two points in space, as a matrix 3 x 2
      The angle is a scalar in rad"""

    # 1. Compute Transformation matrix from the Line and Angle
    g_p1 = line[:, 0].reshape(3, 1)
    g_p2 = line[:, 1].reshape(3, 1)
    line_vector = g_p2 - g_p1
    line_vector_unitary = line_vector / linalg.norm(line_vector)
    delta_line = - g_p1
    t_crossing = np.dot(delta_line.reshape(3), line_vector_unitary.reshape(3))
    g_crossing = t_crossing * line_vector_unitary + g_p1
    v_matrix = np.zeros([3, 3])
    v_matrix[1, 0] = + line_vector_unitary[2]
    v_matrix[2, 0] = - line_vector_unitary[1]
    v_matrix[0, 1] = - line_vector_unitary[2]
    v_matrix[2, 1] = + line_vector_unitary[0]
    v_matrix[0, 2] = + line_vector_unitary[1]
    v_matrix[1, 2] = - line_vector_unitary[0]
    r_matrix = np.identity(3) + np.sin(angle) * v_matrix + (1.0 - np.cos(angle)) * np.matmul(v_matrix, v_matrix)
    d_vector = (np.identity(3) - r_matrix)
    d_vector = np.matmul(d_vector, g_crossing)
    t_matrix = np.zeros([4, 4])
    t_matrix[0:3, 0:3] = r_matrix
    t_matrix[3, 3] = 1.0
    t_matrix[0, 3] = d_vector[0]
    t_matrix[1, 3] = d_vector[1]
    t_matrix[2, 3] = d_vector[2]

    # 2. Rotate the whole trajectory
    dimensions = trajectory_in.shape
    trajectory_out = np.zeros([dimensions[0] + 1, dimensions[1]])
    h_trajectory_in = np.array([0.0, 0.0, 0.0, 1.0]).reshape(4, 1)
    for i in range(dimensions[1]):
        h_trajectory_in[0] = trajectory_in[0, i:(i + 1)]
        h_trajectory_in[1] = trajectory_in[1, i:(i + 1)]
        h_trajectory_in[2] = trajectory_in[2, i:(i + 1)]
        trajectory_out[:, i:(i+1)] = np.matmul(t_matrix, h_trajectory_in)

    # 3. Remove the 4th component of the result, i.e. from homogeneous to normal vector
    trajectory_out = np.delete(trajectory_out, 3, 0)
    return trajectory_out


def extract_data_from_robot_pose(robot_pose):
    """This function extracts the robot position and orientation matrix from a given pose.
        It converts the extracted data to np arrays, a position vector (3x1) and a Rotation matrix (3x3)
            Input:
                    RobotPose of the type given by the library math3d
            Output:
                        Orientation is a 3x3 matrix
                        Position is a 3x1 vector """

    # Init output variables
    orientation = np.zeros([3, 3])
    position = np.zeros([3, 1])

    for i in range(3):
        position[i] = robot_pose.pos[i]
        for j in range(3):
            orientation[j, i] = robot_pose.orient[i][j]

    return orientation, position


def change_robot_position_delta(robot, delta_x=0.0, delta_y=0.0, delta_z=0.0, acc_in=0.01, vel_in=0.01):
    """This function reads out the current pose of the requested Robot and adds the delta vector given by the inputs.
        Then the robot will be moved to that position. The deltas are given in the Global Coordinate System.
            Inputs:
                    Robot: This is the robot number, either 1 or 2
                    deltaX: is the delta to be added to the current x position
                    deltaY: is the deltcreate_smooth_trajectorya to be added to the current y position
                    deltaZ: is the delta to be added to the current z position
            Outputs:
                    No outputs
    """

    pose = robot.get_pose()
    _, position = extract_data_from_robot_pose(pose)

    g_position = transform_from_robot1_to_global(position) if robot.robot_nr == 1 \
        else transform_from_robot2_to_global(position)
    g_position[0] += delta_x
    g_position[1] += delta_y
    g_position[2] += delta_z
    new_position = transform_from_global_to_robot1(g_position) if robot.robot_nr == 1 \
        else transform_from_global_to_robot2(g_position)
    pose.pos[0] = new_position[0]
    pose.pos[1] = new_position[1]
    pose.pos[2] = new_position[2]

    robot.set_pose(pose, acc=acc_in, vel=vel_in)

    return pose


def get_pose_list(robot: urx.Robot,
                  trajectory,
                  steps,
                  robot_nr: int = (1, 2)) -> list:

    pose_list = []

    pose = robot.get_pose()
    orientation, position = extract_data_from_robot_pose(pose)
    g_position = transform_from_robot1_to_global(position) if robot_nr == 1 \
        else transform_from_robot2_to_global(position)

    for i in range(steps):
        x, y, z = trajectory[0, i], trajectory[1, i], trajectory[2, i]
        pose.pos[0] = x
        pose.pos[1] = y
        pose.pos[2] = z
        pose_list.append(pose)
    return pose_list


def change_robot_position_abs(robot: ur3.UR3,
                              robot_nr: int = (1, 2),
                              x: float = 0.0,
                              y: float = 0.0,
                              z: float = 0.0,
                              acc_in: float = 0.01,
                              vel_in: float = 0.01,
                              move_type: str = "movel") -> tuple[float, float, float, float, float, float]:

    """This function reads out the current pose of the requested Robot and changes the x, y and z values to the input
    values.
    Then the robot will be moved to that position. The inputs are given in the Global Coordinate System.
    Inputs:
            robot: This is the robot object
            robot_nr: This is the robot number, either 1 or 2
            x: is the new x value in meters
            y: is the new y value in meters
            z: is the new z value in meters
    Outputs:
            pose: the new pose that the robot moved to
    """

    pose = robot.get_pose()
    _, position = extract_data_from_robot_pose(pose)
    g_position = transform_from_robot1_to_global(position) if robot_nr == 1 \
        else transform_from_robot2_to_global(position)
    g_position[0], g_position[1], g_position[2] = x, y, z

    new_position = transform_from_global_to_robot1(g_position) if robot_nr == 1 \
        else transform_from_global_to_robot2(g_position)
    pose.pos[0] = new_position[0]
    pose.pos[1] = new_position[1]
    pose.pos[2] = new_position[2]
    robot.set_pose(pose, acc=acc_in, vel=vel_in, command=move_type)
    return pose


def change_robot_orientation_delta(robot, delta_alpha=0.0, delta_beta=0.0, delta_gamma=0.0, acc_in=0.01, vel_in=0.01):
    """This function reads out the current pose of the requested Robot and adds the delta angles given by the inputs.
        Then the robot's orientation will be changes. The delta angles are given in the Flange Coordinate System
        All three angles are given in radians.
            Inputs:
                    Robot: This is the robot number, either 1 or 2
                    deltaAlpha: is the delta to be added to the current alpha angle between the orientation and the
                    x-axis
                    deltaBeta: is the delta to be added to the current beta angle between the orientation and the y-axis
                    deltaGamma: is the delta to be added to the current gamma angle of the gripper.
            Outputs:
                    pose: robot pose after orientation change
    """

    pose = robot.get_pose()
    orientation0, _ = extract_data_from_robot_pose(pose)
    z_hat = np.array([0.0, 0.0, 1.0])
    alpha = np.pi / 2 + delta_alpha
    beta = np.pi / 2 + delta_beta
    gamma = np.arccos(np.sqrt(1.0 - np.square(np.cos(alpha)) - np.square(np.cos(beta))))
    v_delta = np.cos(np.array([alpha, beta, gamma]))
    rot_axis = np.cross(z_hat, v_delta)
    rot_axis = rot_axis / linalg.norm(rot_axis)
    rot_angle = np.arccos(np.dot(z_hat, v_delta))
    rot_matrix_delta = compute_rotation_matrix(rot_axis, rot_angle)
    orientation_final = np.matmul(orientation0, rot_matrix_delta)
    rot_matrix_delta_gamma = compute_rotation_matrix(z_hat, delta_gamma)
    orientation_final = np.matmul(orientation_final, rot_matrix_delta_gamma)

    pose.orient = orientation_final
    robot.set_pose(pose, acc=acc_in, vel=vel_in)

    return pose


def change_robot_pose_delta(robot, delta_x=0.0, delta_y=0.0, delta_z=0.0, delta_alpha=0.0, delta_beta=0.0,
                            delta_gamma=0.0, acc_in=0.01, vel_in=0.01):
    """This function is the fusion of two functions:
            - ChangeRobotPositionDelta, and
            - ChangeRobotOrientationDelta.
        This function allows for changes in the entire robot pose at the same time.

        WARNING: The algorithms for the calculation of the inverse kinematics and the interpolator don't work properly
        and do not foresee if the TCP will come close to a singularity. Therefore, it is advisable to use this function
        only in small working spaces.

            Inputs:
                    Same as the inputs of the other two functions.
            Outputs:
                    No outputs
    """
    pose = robot.get_pose()
    orientation0, position = extract_data_from_robot_pose(pose)

    # Compute new orientation matrix
    z_hat = np.array([0.0, 0.0, 1.0])
    alpha = np.pi / 2 + delta_alpha
    beta = np.pi / 2 + delta_beta
    gamma = np.arccos(np.sqrt(1.0 - np.square(np.cos(alpha)) - np.square(np.cos(beta))))
    v_delta = np.cos(np.array([alpha, beta, gamma]))
    rot_axis = np.cross(z_hat, v_delta)
    rot_axis = rot_axis / linalg.norm(rot_axis)
    rot_angle = np.arccos(np.dot(z_hat, v_delta))
    rot_matrix_delta = compute_rotation_matrix(rot_axis, rot_angle)
    orientation_final = np.matmul(orientation0, rot_matrix_delta)
    rot_matrix_delta_gamma = compute_rotation_matrix(z_hat, delta_gamma)
    orientation_final = np.matmul(orientation_final, rot_matrix_delta_gamma)
    pose.orient = orientation_final

    # Compute new position vector
    g_position = transform_from_robot1_to_global(position) if robot.robot_nr == 1 \
        else transform_from_robot2_to_global(position)
    g_position[0] += delta_x
    g_position[1] += delta_y
    g_position[2] += delta_z
    new_position = transform_from_global_to_robot1(g_position) if robot.robot_nr == 1 \
        else transform_from_global_to_robot2(g_position)
    pose.pos[0] = new_position[0]
    pose.pos[1] = new_position[1]
    pose.pos[2] = new_position[2]


    # Execute new pose at the given acceleration and velocity
    robot.set_pose(pose, acc=acc_in, vel=vel_in)

    return pose


def compute_robot_pose_delta(last_pose, robot: ur3.UR3, delta_x=0.0, delta_y=0.0, delta_z=0.0, delta_alpha=0.0, delta_beta=0.0,
                             delta_gamma=0.0):
    """This function computes a pose but doesn't move the robot. It re-utilizes the code of ChangeRobotPoseDelta.
        This function allows for the calculation in the entire robot pose at the same time.

        WARNING: The algorithms for the calculation of the inverse kinematics and the interpolator don't work properly
        and do not foresee if the TCP will come close to a singularity. Therefore, it is advisable to use this function
        only in small working spaces.

            Inputs:
                    Robot
                    deltaX, deltaY and deltaZ
                    deltaAlpha, deltaBeta, deltaGamma

            Outputs:
                    No outputs
    """
    pose = last_pose
    orientation0, position = extract_data_from_robot_pose(pose)

    z_hat = np.array([0.0, 0.0, 1.0])
    alpha = np.pi / 2 + delta_alpha
    beta = np.pi / 2 + delta_beta
    gamma = np.arccos(np.sqrt(1.0 - np.square(np.cos(alpha)) - np.square(np.cos(beta))))
    v_delta = np.cos(np.array([alpha, beta, gamma]))
    rot_axis = np.cross(z_hat, v_delta)
    rot_axis = rot_axis / linalg.norm(rot_axis)
    rot_angle = np.arccos(np.dot(z_hat, v_delta))
    rot_matrix_delta = compute_rotation_matrix(rot_axis, rot_angle)
    orientation_final = np.matmul(orientation0, rot_matrix_delta)
    rot_matrix_delta_gamma = compute_rotation_matrix(z_hat, delta_gamma)
    orientation_final = np.matmul(orientation_final, rot_matrix_delta_gamma)
    pose.orient = orientation_final

    # Compute new position vector
    g_position = transform_from_robot1_to_global(position) if robot.robot_nr == 1 \
        else transform_from_robot2_to_global(position)
    g_position[0] += delta_x
    g_position[1] += delta_y
    g_position[2] += delta_z
    new_position = transform_from_global_to_robot1(g_position) if robot.robot_nr == 1 \
        else transform_from_global_to_robot2(g_position)
    pose.pos[0] = new_position[0]
    pose.pos[1] = new_position[1]
    pose.pos[2] = new_position[2]

    return pose


def rotate_last_joint_delta(robot, delta_gamma=0.0, acc_in=0.01, vel_in=0.01):
    """
    """

    joints = robot.getj()
    joints[5] += delta_gamma
    robot.movej(joints, acc=acc_in, vel=vel_in)


def create_smooth_trajectory(trajectory_orientation, trajectory_position):
    """ This function creates a pose list that can be sent to the "movels" command.
         When executed, the robot will move from point to point according to TrajectoryPosition
         changing the orientation accordingly as given by TrajectoryOrientation without stopping.
            Input:
                    TrajectoryOrientation is an array of matrices X x (3x3); each of these
                     matrices contains the tool orientation, i.e. three parameter orientation.
                    Trajectory Position is an array of vectors 3 x X; each of these
                     vectors contains the TCP, i.e. (x,y,z)
                    Both arrays have the same length and are numpy arrays.
            Output:
                    PoseList is a structured list of length X. Each element of the list
                     is a m3d.Transform, which has two elements, orient and pos. Orient is
                     an array (3x3) and pos is a m3d.Vector"""

    size_of_trajectory = trajectory_orientation.shape[0]
    next_pose = m3d.Transform()
    pose_list = []
    for i in range(size_of_trajectory):
        next_pose.orient = trajectory_orientation[i]
        next_pose.pos = m3d.Vector(trajectory_position[:, i].reshape(3, ))
        pose_list.append(deepcopy(next_pose))
    return pose_list


def create_circle(radius, number_of_points, angle_span):
    """This function create a circular trajectory on the xy-plane depending on the input parameters.
        The first point hast the coordinates (x=Radius, y=0.0, z=0.0)
            Input:
                    Radius (m) is the radius of the circle
                    NumberOfPoints is the total number of points to be calculated
                    AngleSpan (rad) is the total angle to be calculated. 2xPi means the whole
                     circle, but a semicircle can be created if AngleSpan is smaller than 2xPi
            Output:
                    Trajectory_out is a matrix 3xNumberOfPoints containing the circle"""

    # Init Trajectory_out to zeros, z will remain 0.0
    trajectory_out = np.zeros([3, number_of_points])
    delta_angle = angle_span / (number_of_points - 1.0)
    for i in range(number_of_points):
        x = radius * np.cos(delta_angle * i)
        y = radius * np.sin(delta_angle * i)
        trajectory_out[0, i], trajectory_out[1, i] = x, y

    return trajectory_out


def create_helix(radius, radius_rate_of_change, z_final, number_of_points, number_of_turns, offset):
    """This function create a helical trajectory depending on the input parameters.
        The rotation axis is the z-axis.
        The first point hast the coordinates (x=Radius, y=0.0, z=0.0) + Offset
            Input:
                    Radius (m) is the (initial) radius of the helix
                    RadiusRateOfChange is rate for the change of effective radius
                    zSpeed is the rate of change for the z-axis in m
                    NumberOfPoints is the total number of points to be calculated
                    Number of Turns is the total number of turns for the helix. It doesn't have
                     to be an integer
                    Offset is a vector 3x1 that gives an offset for the three coordinates
            Output:
                    Trajectory_out is a matrix 3xNumberOfPoints containing the helix
        The effective radius in the iteration is given by:
            Radius * (1.0 + RadiusRateOfChange * i / NumberOfPoints)
        z grows from 0.0 to zFinal"""

    # Init Trajectory_out to zeros
    trajectory_out = np.zeros([3, number_of_points])
    delta_angle = number_of_turns * 2 * np.pi / (number_of_points - 1.0)
    for i in range(number_of_points):
        effective_radius = radius * (1.0 + radius_rate_of_change / number_of_points * i)
        x = effective_radius * np.cos(delta_angle * i) + offset[0]
        y = effective_radius * np.sin(delta_angle * i) + offset[1]
        z = z_final / number_of_points * i + offset[2]
        trajectory_out[0, i], trajectory_out[1, i], trajectory_out[2, i] = x, y, z

    return trajectory_out


def config_robot_2(robot, move_to_home_position=False, speed='half'):
    # Todo - remove unneeded set-commands
    home_position_joints = [0.0, -2.240720, 1.172209, -1.466363, 0.0, 0.604892]
    home_position_joints1 = np.array(home_position_joints)
    home_position_joints2 = mirror_joints1to2(home_position_joints1)

    print(' Robot2: Configuring')

    # TCP must be defined using the original coordinate system of the robot flange
    # Gripper (RG2 new) is 2 cm longer than the last version
    # Instrument - needle holder - has a length of 69 mm
    robot.set_tcp((0.0, 0.0, + 0.26 + 0.02 + 0.069 + 0.005, 0.0, 0.0, 0.0))
    robot.set_tcp((0.0, 0.0, + 0.26 + 0.02 + 0.069 - 0.005, 0.0, 0.0, 0.0))
    robot.set_tcp((0.0, 0.0, + 0.26 + 0.069, 0.0, 0.0, 0.0))
    # Calibrated on 2021-08.05
    robot.set_tcp((+ 0.00126, - 0.00253, + 0.33689, - 0.0043, - 0.0282, + 0.0301))

    # Calibrated on 2021-08.06
    # Todo - Find out why there is a rob1 command in the config function for rob2
    # rob1.set_tcp((+ 0.00131, + 0.00923, + 0.30426, - 0.0043, - 0.0282, 0.0301))

    # Gripper (RG2 new) weight is 0.79 kg
    robot.set_payload(+ 0.79, (0.0, 0.0, 0.1))

    # Force of gravity must be defined according to the mounting,
    # i.e. gravity works against y-axis
    robot.set_gravity((0.0, + 9.81, 0.0))

    if move_to_home_position:
        print('     Robot2: Moving to HomePosition out of singularities')
        if speed == 'full':
            velocity = v_joints
        elif speed == 'half':
            velocity = v_joints / 2.0
        else:
            velocity = v_joints

        robot.movej(home_position_joints2, a_joints, velocity)


def calculate_circle_points_with_rotation_for_robot_1(robot, diameter, angles=(0, 90, 180, 270)):
    pose = robot.get_pose()
    orient, coords = extract_data_from_robot_pose(pose)

    rotation_matrix = orient
    v1 = rotation_matrix[:, 0]
    v2 = rotation_matrix[:, 1]

    center = transform_from_robot1_to_global(coords)

    center[2] -= diameter / 2
    center = transform_from_global_to_robot1(center)

    points_on_circle = []
    for angle in angles:
        theta = np.radians(angle)
        circle_point = center[0] + diameter / 2 * (np.cos(theta) * v1 + np.sin(theta) * v2)
        points_on_circle.append(circle_point)

    pose_1 = pose.pos
    pose_1.pos = points_on_circle[0]
    pose_2 = pose.pos
    pose_2.pos = points_on_circle[0]
    pose_3 = pose.pos
    pose_3.pos = points_on_circle[0]
    pose_4 = pose.pos
    pose_4.pos = points_on_circle[0]

    return [pose_1, pose_2, pose_3, pose_4]


def perform_circular_motion(robot, diameter):
    """
    This function makes the robot perform a circular motion on a horizontal plane (x ynd y plane) relative to the tcp.
    This can be used to perform the motion to tie the knot
    Parameters:
        robot: the Robot to perform the motion
        diameter: the diameter of the circular motion in mm?
    Outputs:
        None
    """
    robot.x_t = -(diameter / 2)
    robot.y_t = -(diameter / 2)
    robot.x_t = diameter / 2
    robot.y_t = diameter / 2


def needle_control(robot):
    # Todo - Find out what this function is used for
    j2 = np.array([-29.42031949, -119.03606238,  -22.87265427, -127.46992533, 90.10265342,  -27.7523873]) * np.pi / 180
    robot.movej(j2, a_joints, v_joints)


def create_transformation_matrices():
    """
    robot1_th_matrix = np.array([[-3.71890808e-03, 9.99993061e-01, -2.18120375e-04, -2.82526220e-01],
                                 [7.94638441e-03, 2.47667074e-04, 9.99968396e-01, -5.71474276e-01],
                                 [9.99961512e-01, 3.71705729e-03, -7.94725032e-03, 4.10260571e-01],
                                 [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    robot1_th_matrix = np.array([[-0.01785007,  0.99983351, -0.00378596, -0.28272082],
                                  [ 0.01118036,  0.00398593,  0.99992955, -0.57181916],
                                  [ 0.99977816,  0.01780648, -0.01124965,  0.41203405],
                                  [ 0.        ,  0.        ,  0.        ,  1.        ]])


    robot1_th_matrix = np.array([[-0.55726233,  0.83032998, -0.00328932, -0.27528852],
                                 [ 0.00600828,  0.00799361,  0.99995,    -0.5717007 ],
                                 [ 0.83031476,  0.55721471, -0.00944339,  0.41887809],
                                 [ 0.,          0.,          0.,          1.        ]])
    """


    robot1_th_matrix = np.array([[-0.0024844,   0.99999497, -0.0019698,  -0.28167977],
                                 [ 0.00833202,  0.00199044,  0.99996331, -0.57187498],
                                 [ 0.9999622,   0.0024679,  -0.00833692,  0.41210053],
                                 [ 0.,          0.,          0.,          1.        ]]
)

    robot2_th_matrix = np.array([[-0.001654, 0.99986498, 0.01634858, 0.26070242],
                                 [0.00345292, -0.01634279, 0.99986049, -0.5759687],
                                 [0.99999267, 0.00171022, -0.00342542, 0.40778839],
                                 [0.0, 0.0, 0.0, 1.0]])

    return robot1_th_matrix, robot2_th_matrix


def create_inverse_transformation_matrices():
    """
    New Transformation function after implementation of kabsch-algorithm
    """
    transformation_matrix_1, transformation_matrix_2 = create_transformation_matrices()

    def create_single_transformation_matrix(transformation_matrix):
        robot_th_matrix_inv = np.eye(4)
        robot_rotation_inv = np.transpose(transformation_matrix[0:3, 0:3])
        robot_translation_vector = transformation_matrix[0:3, 3]
        robot_translation_vector_inv = np.matmul(- robot_rotation_inv, robot_translation_vector)
        robot_th_matrix_inv[0:3, 0:3] = robot_rotation_inv
        robot_th_matrix_inv[0:3, 3] = robot_translation_vector_inv
        return robot_th_matrix_inv

    robot1_th_matrix_inv = create_single_transformation_matrix(transformation_matrix_1)
    robot2_th_matrix_inv = create_single_transformation_matrix(transformation_matrix_2)

    return robot1_th_matrix_inv, robot2_th_matrix_inv


def calibrate_tcp(robot: ur3.UR3, delta_translation=0.0005, delta_orientation=0.01):
    """
    Result of calibration for robot 1 on 20240430:
    <Transform:
    <Orientation:
    array([[-0.99839804,  0.05650037, -0.0030104 ],
           [ 0.00234374, -0.01186178, -0.9999269 ],
           [-0.05653195, -0.99833211,  0.01171036]])>
    <Vector: (-0.28307, -0.57542, 0.41057)>
    """
    read_input = True
    while read_input:
        pose = robot.get_pose()

        _, position = extract_data_from_robot_pose(pose)
        if robot.robot_nr == 1:
            print("Global Pose vector: ", transform_from_robot1_to_global(position))
        if robot.robot_nr == 2:
            print("Global Pose vector: ", transform_from_robot2_to_global(position))
        command = input("direction x/y/z: ")

        if command == "x+":
            change_robot_position_delta(robot, delta_translation, 0.000, 0.000, 1, 1)
        elif command == "y+":
            change_robot_position_delta(robot, 0.000, delta_translation, 0.000, 1, 1)
        elif command == "z+":
            change_robot_position_delta(robot, 0.000, 0.000, delta_translation, 1, 1)
        elif command == "x-":
            change_robot_position_delta(robot, -delta_translation, 0.000, 0.000, 1, 1)
        elif command == "y-":
            change_robot_position_delta(robot, 0.000, -delta_translation, 0.000, 1, 1)
        elif command == "z-":
            change_robot_position_delta(robot, 0.000, 0.000, -delta_translation, 1, 1)
        elif command == "a+":
            change_robot_orientation_delta(robot, delta_orientation, 0.000, 0.000, 1, 1)
        elif command == "b+":
            change_robot_orientation_delta(robot, 0.000, delta_orientation, 0.000, 1, 1)
        elif command == "g+":
            change_robot_orientation_delta(robot, 0.000, 0.000, delta_orientation, 1, 1)
        elif command == "a-":
            change_robot_orientation_delta(robot, -delta_orientation, 0.000, 0.000, 1, 1)
        elif command == "b-":
            change_robot_orientation_delta(robot, 0.000, -delta_orientation, 0.000, 1, 1)
        elif command == "g-":
            change_robot_orientation_delta(robot, 0.000, 0.000, -delta_orientation, 1, 1)
        elif command == "q":
            read_input = False
    if robot.robot_nr == 1:
        print("Global Pose vector: ", transform_from_robot1_to_global(position))
    if robot.robot_nr == 2:
        print("Global Pose vector: ", transform_from_robot2_to_global(position))


def fine_tune_tcp(robot: ur3.UR3, delta_translation=0.0005, delta_orientation=0.01):
    command_map = {
        "x+": (delta_translation, 0.000, 0.000, 0, 0, 0),
        "x-": (-delta_translation, 0.000, 0.000, 0, 0, 0),
        "y+": (0.000, delta_translation, 0.000, 0, 0, 0),
        "y-": (0.000, -delta_translation, 0.000, 0, 0, 0),
        "z+": (0.000, 0.000, delta_translation, 0, 0, 0),
        "z-": (0.000, 0.000, -delta_translation, 0, 0, 0),
        "a+": (0, 0, 0, delta_orientation, 0, 0),
        "a-": (0, 0, 0, -delta_orientation, 0, 0),
        "b+": (0, 0, 0, 0, delta_orientation, 0),
        "b-": (0, 0, 0, 0, -delta_orientation, 0),
        "g+": (0, 0, 0, 0, 0, delta_orientation),
        "g-": (0, 0, 0, 0, 0, -delta_orientation)
    }

    while True:
        command = input(
            "Input direction x/y/z or a/b/g for orientation (e.g., 'x+', 'y-', 'a+', 'g-'). 'q' to quit: ").strip()

        if command == "q":
            break

        if command in command_map:
            dx, dy, dz, da, db, dg = command_map[command]
            if dx != 0 or dy != 0 or dz != 0:
                change_robot_position_delta(robot, dx, dy, dz, 1, 1)
            if da != 0 or db != 0 or dg != 0:
                change_robot_orientation_delta(robot, da, db, dg, 1, 1)
    pose = robot.get_pose()
    _, position = extract_data_from_robot_pose(pose)
    if robot.robot_nr == 1:
        print(f"Robot 1 Global position after fine tuning: \n {transform_from_robot1_to_global(position)}")
    else:
        print(f"Robot 2 Global position after fine tuning: \n {transform_from_robot2_to_global(position)}")

def get_joint_angles(robot):
    while True:
        inp = input("Press Enter to get joints. Press 'q' to quit")
        if inp == 'q':
            return
        print(robot.getj())


def calculate_needle_trajectory(a, b, theta, plot=False):

    def calculate_r(a, b, phi):
        r_squared = (a**2 * b*+2) / (a**2 * (sin(phi))**2 + b**2 * (cos(phi))**2)
        return sqrt(r_squared)

    def calculate_r_dot(a, b, phi):
        r_dot = (a * b * (b**2 - a**2) * sin(2*phi)) / (2 * ((a**2 * (sin(phi))**2 + b**2 * (cos(phi))**2)**(3/2)))
        return r_dot

    def calculate_orientation_vector(r_dot, r, phi):
        y_hat = np.array([0, 1, 0])
        z_hat = np.array([0, 0, 1])
        o_vec = y_hat * (r_dot * cos(phi) - r * sin(phi)) + z_hat * (r_dot * sin(phi) + r * cos(phi))
        magnitude = np.linalg.norm(o_vec)

        if magnitude != 0:
            normalized_o_vec = o_vec / magnitude
        else:
            normalized_o_vec = o_vec

        return o_vec, normalized_o_vec

    def plot_vectors(vectors, o_vectors):
        fig, ax = plt.subplots()

        ys = [vec[1] for vec in vectors]
        zs = [vec[2] for vec in vectors]

        oys = [o_vec[1] for o_vec in o_vectors]
        ozs = [o_vec[2] for o_vec in o_vectors]

        for y, z, oy, oz in zip(ys, zs, oys, ozs):
            ax.quiver(0, 0, y, z, color='b', angles='xy', scale_units='xy', scale=1)
            ax.quiver(y, z, oy, oz, color='g', angles='xy', scale_units='xy', scale=1)

        ax.set_xlabel('Y')
        ax.set_ylabel('Z')

        ax.set_aspect('equal', 'box')
        plt.xlim(-5, 5)
        plt.ylim(-3, 3)

        plt.grid(True)
        plt.show()

    vectors = []
    o_vectors = []

    for phi_deg in range(theta, 0+10, 10):
        phi = radians(phi_deg)
        r = calculate_r(a, b, phi)
        r_dot = calculate_r_dot(a, b, phi)
        o_vec, normalized_o_vec = calculate_orientation_vector(r_dot, r, phi)
        y = r * cos(phi)
        z = r * sin(phi)
        vec = np.array([0, y, z])
        vectors.append(vec)
        o_vectors.append(normalized_o_vec)

    if plot:
        plot_vectors(vectors, o_vectors)

    print(vectors, o_vectors)
    return vectors, o_vectors


if __name__ == '__main__':
    # robot, _ = suture_process.initialize_robots(True, False, init_home_pos=False)
    # calibrate_tcp(robot)
    create_transformation_matrices()

