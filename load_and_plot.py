import numpy as np
import matplotlib.pyplot as plt
import os

def geodhesic_error(R1, R2):
    """
    Calculate the geodesic error between two rotation matrices R1 and R2.
    The geodesic error is defined as the angle of the rotation that takes R1 to R2.
    
    Parameters:
    R1 : np.ndarray
        A 3x3 rotation matrix.
    R2 : np.ndarray
        A 3x3 rotation matrix.
        
    Returns:
    float
        The geodesic error in radians.
    """
    # Compute the relative rotation matrix
    R_rel = np.dot(R1.T, R2)
    
    # Compute the angle of rotation using the trace of the relative rotation matrix
    angle = np.arccos((np.trace(R_rel) - 1) / 2)
    
    return angle

os.chdir("/mnt/c/Users/mougo/Downloads/rl_control_armrobot_labor/rl_control_armrobot")
#os.chdir(os.path.dirname(os.path.abspath(__file__)))
robot = True
a=5; b=3
# np.savetxt("cross_error_vecs.csv", cross_error_vecs, delimiter=",")
# np.savetxt("ori_errors.csv", ori_errors, delimiter=",")

positions = np.load("./positions.npy") #loaded_arr = np.load("my_3d_array.npy")
orientations = np.load("./orientations.npy")
cross_error_vecs = np.load("./cross_error_vecs.npy")
ori_errors = np.rad2deg(np.arccos(np.load("./ori_errors.npy")))

#np.savetxt("positions.csv", positions, delimiter=",")
#np.savetxt("orientations.csv", orientations, delimiter=",")

if robot:
    positions_rob = np.load("./positions_rob.npy")
    orientations_rob = np.load("./orientations_rob.npy")
    cross_error_vecs_rob = np.load("./cross_error_vecs_rob.npy")
    ori_errors_rob = np.rad2deg(np.arccos(np.load("./ori_errors_rob.npy")))
    position_rob = positions_rob[-1]
    # np.savetxt("cross_error_vecs_rob.csv", cross_error_vecs_rob, delimiter=",")
    # np.savetxt("ori_errors_rob.csv", ori_errors_rob, delimiter=",")
position = positions[-1]
    
print("pos shape: ", position)
    # np.save("positions_rob.npy", positions_rob) #loaded_arr = np.load("my_3d_array.npy")
    # np.save("orientations_rob.npy", orientations_rob)
    # np.save("cross_error_vecs_rob.npy", cross_error_vecs_rob)
    # np.save("ori_errors_rob.npy", ori_errors_rob)
mean_squared_errors = [np.mean(np.array(cross_error_vec)**2) for cross_error_vec in cross_error_vecs]
root_mean_squared_errors = [np.sqrt(mse) for mse in mean_squared_errors]
if robot:
    mean_squared_errors_rob = [np.mean(np.array(cross_error_vec_rob)**2) for cross_error_vec_rob in cross_error_vecs_rob]
    root_mean_squared_errors_rob = [np.sqrt(mse) for mse in mean_squared_errors_rob]

mean_squared_errors_ori = [np.mean(np.array(ori_error)**2) for ori_error in ori_errors]
root_mean_squared_errors_ori = [np.sqrt(mse) for mse in mean_squared_errors_ori]
if robot:
    mean_squared_errors_ori_rob = [np.mean(np.array(ori_error_rob)**2) for ori_error_rob in ori_errors_rob]
    root_mean_squared_errors_ori_rob = [np.sqrt(mse) for mse in mean_squared_errors_ori_rob]

# print(orientations[0])
# print(orientations_rob[0])
# meas_ori_shifts=[]
# for i in range(len(orientations)):
#     meas_ori_shift = [np.rad2deg(geodhesic_error(ori, ori_rob)) for ori, ori_rob in zip(orientations[i], orientations_rob[i])] if robot else None
#     meas_ori_shifts.append(meas_ori_shift)

fig = plt.figure(figsize=(18, 6))

# if meas_ori_shifts is not None:
#     ax0 = fig.add_subplot(1, 1, 1)
#     #ax0.plot(list(range(1,len(meas_ori_shift)+1)), meas_ori_shift, 'b', linewidth=2, label='Geodesic Error')
#     ax0.boxplot(list(meas_ori_shifts), showfliers=True, flierprops=dict(marker='o', markerfacecolor='orange', markeredgecolor='orange', markersize=5))
#     ax0.set_title(f'Geodesic Error zwischen Agent und echten Roboter Orientation along the Half-Ellipse mit a={a}mm, b={b}mm', fontsize=11, fontweight='bold')
#     ax0.set_xlabel('Episodes', fontsize=9, fontweight='bold')
#     ax0.set_ylabel('Error(°)', fontsize=9, fontweight='bold')
#     ax0.grid(True)
#     plt.tight_layout()
#     plt.ion()
#     plt.legend()
#     plt.show(block=True)   

ax1 = fig.add_subplot(1, 3, 1, projection='3d')
# plot agent's path
ax1.set_title(f'TCP-Positionen entlang der Halbellipse mit a={a}mm, b={b}mm', fontsize=11, fontweight='bold')
ax1.set_xlabel('x_t (mm)', fontsize=9, fontweight='bold')
ax1.set_ylabel('y_t (mm)', fontsize=9, fontweight='bold')
ax1.set_zlabel('z_t (mm)', fontsize=9, fontweight='bold')
ax1.plot3D(position[:, 0], position[:, 1], position[:, 2], 'b', linewidth=2)
ax1.scatter3D(position[:, 0], position[:, 1], position[:, 2], 'b', linewidth=2, label="Agent TCP Path")
if robot:
    ax1.plot3D(position_rob[:, 0], position_rob[:, 1], position_rob[:, 2], 'orange', linewidth=2)
    ax1.scatter3D(position_rob[:, 0], position_rob[:, 1], position_rob[:, 2], 'orange', linewidth=2, label="Robot TCP Path")
# ax1.quiver(position[:, 0], position[:, 1], position[:, 2], O_tcp[0, :], O_tcp[1, :], O_tcp[2, :], color='r', length=0.3, label="TCP Orientation")
# ax1.quiver(position[:, 0], position[:, 1], position[:, 2], O_tcp_int[0, :], O_tcp_int[1, :], O_tcp_int[2, :], color='black', length=0.3, label="TCP Orientation Target")
# if robot:
#     ax1.quiver(position_rob[:, 0], position_rob[:, 1], position_rob[:, 2], O_tcp_rob[0, :], O_tcp_rob[1, :], O_tcp_rob[2, :], color='orange', length=0.3, label="Robot TCP Orientation")
#     ax1.quiver(position_rob[:, 0], position_rob[:, 1], position_rob[:, 2], O_tcp_int_rob[0, :], O_tcp_int_rob[1, :], O_tcp_int_rob[2, :], color='black', length=0.3, label="Robot TCP Orientation Target")
ax1.set_box_aspect((10, 8, 5))
# print(mean_squared_errors)
ax2 = fig.add_subplot(1, 3, 2)#, projection='3d')

ax2.plot(list(range(1,len(mean_squared_errors)+1)), mean_squared_errors, 'b', linewidth=2, label='MSE Agent')
#ax2.scatter(list(range(len(mean_squared_errors))), mean_squared_errors, 'b', linewidth=2)
ax2.plot(list(range(1,len(root_mean_squared_errors)+1)), root_mean_squared_errors, 'b--', linewidth=2, label='RMSE Agent')
#ax2.scatter(list(range(len(root_mean_squared_errors))), root_mean_squared_errors, 'b--', linewidth=2)
ax2.boxplot(list(cross_error_vecs), showfliers=True, flierprops=dict(marker='o', markerfacecolor='orange', markeredgecolor='orange', markersize=5))
plt.legend()
if robot:
    ax2.plot(list(range(1,len(mean_squared_errors_rob)+1)), mean_squared_errors_rob, 'orange', linewidth=2, label='MSE Robot')
    ax2.plot(list(range(1,len(root_mean_squared_errors_rob)+1)), root_mean_squared_errors_rob, 'o--', linewidth=2, label='RMSE Robot')

ax3 = fig.add_subplot(1, 3, 3)#, projection='3d')
ax3.plot(list(range(1,len(mean_squared_errors_ori)+1)), mean_squared_errors_ori, 'b', linewidth=2, label='MSE Ori Agent')
ax3.plot(list(range(1,len(root_mean_squared_errors_ori)+1)), root_mean_squared_errors_ori, 'b--', linewidth=2, label='RMSE Ori Agent')
ax3.boxplot(list(ori_errors), flierprops=dict(marker='o', markerfacecolor='orange', markeredgecolor='orange', markersize=5))
if robot:
    ax3.plot(list(range(1,len(mean_squared_errors_ori_rob)+1)), mean_squared_errors_ori_rob, 'orange', linewidth=2, label='MSE Ori Robot')
    ax3.plot(list(range(1,len(root_mean_squared_errors_ori_rob)+1)), root_mean_squared_errors_ori_rob, 'o--', linewidth=2, label='RMSE Ori Robot')

# ax2.plot(list(range(max([len(vec) for vec in cross_error_vecs]))), [0.05]*max([len(vec) for vec in cross_error_vecs]), 'r--', linewidth=3)
# ax3.plot(list(range(max([len(vec) for vec in ori_errors]))), [1.5]*max([len(vec) for vec in ori_errors]), 'r--', linewidth=3)

# for i, cross_error_vec in enumerate(cross_error_vecs):
#     #ax2.scatter(list(range(len(cross_error_vec))), cross_error_vec, linewidth=1)
#     ax2.plot(list(range(len(cross_error_vec))), cross_error_vec, linewidth=1, label=f'eps {i+1}')
    
#     # for j in range(len(cross_error_vec)):
#     #     ax2.text(j, cross_error_vec[j], f'{cross_error_vec[j]:.2f}', fontsize=8)
#     #ax2.text(list(range(len(cross_error_vec))), cross_error_vec,[f'{x:.2f}' for x in cross_error_vec])
#     #ax2.quiver(P_tcp[0, :], P_tcp[1, :], P_tcp[2, :], O_tcp[0, :], O_tcp[1, :], O_tcp[2, :], color='r', length=0.2)
# ax2.set_title('Target tcp cross error\n(base position)')
# ax2.set_xlabel('steps')
# ax2.set_ylabel('error(mm)')
#ax2.set_zlabel('z_tcp')
ax2.set_title(f'TCP-Positionsfehler entlang der Halbellipse mit a={a}mm, b={b}mm', fontsize=11, fontweight='bold')
ax2.set_xlabel('Episoden', fontsize=9, fontweight='bold')
ax2.set_ylabel('Error(mm)', fontsize=9, fontweight='bold')
ax3.set_title(f'TCP-Orientierungsfehler entlang der Halbellipse mit a={a}mm, b={b}mm', fontsize=11, fontweight='bold')
ax3.set_xlabel('Episoden', fontsize=9, fontweight='bold')
ax3.set_ylabel('Error(°)', fontsize=9, fontweight='bold')
ax2.grid(True)
ax3.grid(True)
plt.tight_layout()
plt.ion()
plt.legend()
plt.show(block=True)    
print(positions)
# print("Done! You can now watch test_env4.mp4")
# 5. Clean up
#env.close()
print("Testing complete.")