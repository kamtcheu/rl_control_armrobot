import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# 1. Geometry and Configuration Constants
# -------------------------------------------------------------------------
a = 5 # 15 mm in meters
b = 3# 10 mm in meters
k = (a**2 - b**2) / a**2
Rn = 12 
theta_h = +90 / 180 * np.pi

# Note: Keeping your large numbers intact from the MATLAB script
elipse_center_global = np.array([[-289.997], [+6.666], [+247.185]])  # Convert mm to meters

N = 19  # Number of points to generate, matching MATLAB's 180:10:360
# theta = np.linspace(180, 360, N) * np.pi / 180  # Convert degrees to radians
# theta_start = theta[0]
# theta_end = theta[-1]
# -------------------------------------------------------------------------
# 2. Homogeneous Transformation Matrices
# -------------------------------------------------------------------------
# Global to Base
R_global2base = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]
])
# R_global2base = np.array([
#     [1, 0, 0],
#     [0, 1, 0],
#     [0, 0, 1]
# ]).T
d_global2base =  np.array([[-405], [-250], [+555]])  # Convert mm to meter

T_global2base = np.eye(4)
T_global2base[0:3, 0:3] = R_global2base
T_global2base[0:3, 3:4] = -R_global2base @ d_global2base

# Base to Global
R_base2global = R_global2base.T
#d_base2global = -R_base2global @ d_global2base

T_base2global = np.eye(4)
T_base2global[0:3, 0:3] = R_base2global
T_base2global[0:3, 3:4] = d_global2base

# TCP to T : active interpretation of the transformation, moves the point from TCP to T inside the same "base" frame
R_tcp2t = np.array([
    [1, 0, 0],
    [0, np.cos(theta_h - np.pi / 2), -np.sin(theta_h - np.pi / 2)],
    [0, np.sin(theta_h - np.pi / 2),  np.cos(theta_h - np.pi / 2)]
])
d_tcp2t = np.array([[0], [-Rn * np.sin(theta_h)], [Rn * (np.cos(theta_h) - 1)]])

T_tcp2t = np.eye(4)
T_tcp2t[0:3, 0:3] = R_tcp2t
T_tcp2t[0:3, 3:4] = d_tcp2t

# T to TCP : active interpretation of the transformation, moves the point from T to TCP inside the same "base" frame
R_t2tcp = R_tcp2t.T
d_t2tcp = -R_tcp2t @ d_tcp2t

T_t2tcp = np.eye(4)
T_t2tcp[0:3, 0:3] = R_t2tcp
T_t2tcp[0:3, 3:4] = d_t2tcp

# Constant Rotation Matrix (X-axis -90 deg)
R_y_90 = np.array([
    [np.cos(np.pi / 2), 0, np.sin(np.pi / 2)],
    [0, 1, 0],
    [-np.sin(np.pi / 2), 0, np.cos(np.pi / 2)]
])
R_x_m90 = np.array([
    [1, 0, 0],
    [0, np.cos(-np.pi / 2), -np.sin(-np.pi / 2)],
    [0, np.sin(-np.pi / 2),  np.cos(-np.pi / 2)]
])
# -------------------------------------------------------------------------
# 3. Angle discretization & Buffer Init
# -------------------------------------------------------------------------
phi_deg = np.linspace(180, 360, N) # np.arange(180, 361, 10)  # Generates 19 elements matching 180:10:360
phi = phi_deg * np.pi / 180
delta_z = 0.0

# Preallocate result matrices (3 rows, 19 columns)
P_t = np.zeros((3, N))
O_t = np.zeros((3, N))
O_t2tcp = np.zeros((3, N))
P_tcp = np.zeros((3, N))
O_tcp = np.zeros((3, N))
P_base = np.zeros((3, N))
O_base = np.zeros((3, N))
Nor = np.zeros((3, N))
T = np.zeros((3, N))
BiNor = np.zeros((3, N))

P_t_base = np.zeros((3, N))
O_t_base = np.zeros((3, N))

# -------------------------------------------------------------------------
# 4. Main Computation Loop
# -------------------------------------------------------------------------
for i in range(N):
    # Radius computation
    P_r_val = np.sqrt(b**2 / (1 - k * (np.cos(phi[i]))**2))
    
    # Target position
    P_t_col = np.array([[0.0], [P_r_val * np.cos(phi[i])], [P_r_val * np.sin(phi[i])]])
    P_t[:, i : i + 1] = P_t_col + elipse_center_global + np.array([[0], [0], [delta_z]])

    # Tangent vector derivative logic
    P_r_dot_val = (-0.5 * b * np.sin(2 * phi[i]) * k) / (1 - k * (np.cos(phi[i]))**2)**1.5
    
    O_t_col = np.array([
        [0],
        [-P_r_val * np.sin(phi[i]) + P_r_dot_val * np.cos(phi[i])],
        [+P_r_val * np.cos(phi[i]) + P_r_dot_val * np.sin(phi[i])]
    ])
    O_t[:, i : i + 1] = (O_t_col / np.linalg.norm(O_t_col))
    P_t_homogen = np.append(P_t[:, i], 1.0).reshape(4, 1)
    P_tcp_homogen = T_t2tcp @ P_t_homogen
    P_tcp[:, i] = P_tcp_homogen[0:3].flatten()

    # O_t_homogen = np.append(O_t[:, i], 1.0).reshape(4, 1)
    # O_tcp_homogen = T_t2tcp @ O_t_homogen
    # O_tcp[:, i] = O_tcp_homogen[0:3].flatten()
    O_tcp[:, i] = (R_x_m90 @ R_t2tcp @ O_t[:, i : i + 1]).flatten()
    O_t2tcp[:, i] = (R_t2tcp @ O_t[:, i : i + 1]).flatten()

    # Homogeneous transformations to Robot Base space
    P_base_homogen = T_global2base @ P_tcp_homogen
    P_base[:, i] = P_base_homogen[0:3].flatten()
    O_base[:, i] = (R_global2base @ O_tcp[:, i : i + 1]).flatten()


    P_t_base_homogen = T_global2base @ P_t_homogen
    P_t_base[:, i] = P_t_base_homogen[0:3].flatten()
    O_t_base[:, i] = (R_global2base @ O_t[:, i : i + 1]).flatten()
    
    # Matching MATLAB index criteria (1-based > 10 maps to 0-based >= 10)
    if i >= 10:
        delta_z += 0 * 0.2
T = O_t2tcp#np.array([O_t2tcp[0, :], O_t2tcp[1, :], O_t2tcp[2, :]]) - np.array([P_tcp[0, :], P_tcp[1, :], P_tcp[2, :]])
BiNor = O_tcp# np.array([O_tcp[0, :], O_tcp[1, :], O_tcp[2, :]]) - np.array([P_tcp[0, :], P_tcp[1, :], P_tcp[2, :]])# np.cross(T, N, axis=0)
N = np.cross([T[0, :], T[1, :], T[2, :]], [BiNor[0, :], BiNor[1, :], BiNor[2, :]], axis=0)

T_base = O_t_base
BiNor_base = O_base
N_base = np.cross([T_base[0, :], T_base[1, :], T_base[2, :]], [BiNor_base[0, :], BiNor_base[1, :], BiNor_base[2, :]], axis=0)
N_base = N_base / np.linalg.norm(N_base, axis=0)  # Normalize the normal vector
#N = np.array([np.cross([T[0, i:i+1], T[1, i:i+1], T[2, i:i+1]], [BiNor[0, i:i+1], BiNor[1, i:i+1], BiNor[2, i:i+1]], axis=0) for i in range(N)])
#N = np.cross([T[0, 0], T[1, 0], T[2, 0]], [BiNor[0, 0], BiNor[1, 0], BiNor[2, 0]]) #np.array([[1],[0],[0]])
N = N / np.linalg.norm(N, axis=0)  # Normalize the normal vector
#print(np.dot(T[:, 0], N[:, 0], ))
print("ecllipse_center_base: ", T_global2base @ np.append(elipse_center_global, 1.0).reshape(4, 1))
# -------------------------------------------------------------------------
# 5. Plotting Generation (3 Subplots matching MATLAB figure 1)
# -------------------------------------------------------------------------
fig = plt.figure(figsize=(18, 6))

# Subplot 1: Target Needle Tip
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.plot3D(P_t[0, :], P_t[1, :], P_t[2, :], 'b', linewidth=2)
ax1.quiver(P_t[0, :], P_t[1, :], P_t[2, :], O_t[0, :], O_t[1, :], O_t[2, :], color='r', length=0.2)
ax1.set_title('Target Needle Tip\n(global pos + global orient)')
ax1.set_xlabel('x_t')
ax1.set_ylabel('y_t')
ax1.set_zlabel('z_t')
#ax1.view_init(elev=90, azim=0) # Emulates view(90,0)
ax1.grid(True)

# Subplot 2: Target TCP
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.plot3D(P_t[0, :], P_t[1, :], P_t[2, :], 'b', linewidth=2)
ax2.quiver(P_t[0, :], P_t[1, :], P_t[2, :], O_t[0, :], O_t[1, :], O_t[2, :], color='r', length=0.2)
ax2.plot3D(P_tcp[0, :], P_tcp[1, :], P_tcp[2, :], 'y', linewidth=2)
ax2.quiver(P_tcp[0, :], P_tcp[1, :], P_tcp[2, :], O_t[0, :], O_t[1, :], O_t[2, :], color='r', length=0.2)
ax2.quiver(P_tcp[0, :], P_tcp[1, :], P_tcp[2, :], O_tcp[0, :], O_tcp[1, :], O_tcp[2, :], color='g', length=0.2)
ax2.quiver(P_tcp[0, :], P_tcp[1, :], P_tcp[2, :], N[0, :], N[1, :], N[2, :], color='b', length=0.2)
ax2.set_title('Target tcp\n(global position)')
ax2.set_xlabel('x_tcp')
ax2.set_ylabel('y_tcp')
ax2.set_zlabel('z_tcp')
ax2.grid(True)

# Subplot 3: Required Base Position
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.plot3D(P_base[0, :], P_base[1, :], P_base[2, :], 'b', linewidth=2)
ax3.quiver(P_base[0, :], P_base[1, :], P_base[2, :], O_base[0, :], O_base[1, :], O_base[2, :], color='r', length=0.2)
ax3.quiver(P_base[0, :], P_base[1, :], P_base[2, :], O_t_base[0, :], O_t_base[1, :], O_t_base[2, :], color='g', length=0.2)
ax3.quiver(P_base[0, :], P_base[1, :], P_base[2, :], N_base[0, :], N_base[1, :], N_base[2, :], color='y', length=0.2)
ax3.set_title('Required Base\nposition and orientation')
ax3.set_xlabel('x_base')
ax3.set_ylabel('y_base')
ax3.set_zlabel('z_base')
ax3.grid(True)

plt.tight_layout()
plt.ion()
plt.show(block=True)
#%%

#%%
import numpy as np
import matplotlib.pyplot as plt
def elliptical_trajectory(a=5, b=3, Rn=12, theta_h=90/180*np.pi, elipse_center_global=np.array([[-289.997], [+6.666], [+247.185]]), N=19, show_plot=False):
    """
    Generates an elliptic trajectory for a robotic arm based on predefined parameters.
    
    Returns:
        P_base (np.ndarray): The positions in the robot base frame.
        O_base (np.ndarray): The orientations in the robot base frame.
    """
    k = (a**2 - b**2) / a**2

    # -------------------------------------------------------------------------
    # 2. Homogeneous Transformation Matrices
    # -------------------------------------------------------------------------
    # Global to Base
    R_global2base = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])
    # R_global2base = np.array([
    #     [1, 0, 0],
    #     [0, 1, 0],
    #     [0, 0, 1]
    # ])
    d_global2base =  np.array([[-405], [-250], [+555]])  # Convert mm to meters

    T_global2base = np.eye(4)
    T_global2base[0:3, 0:3] = R_global2base
    T_global2base[0:3, 3:4] = -R_global2base @ d_global2base
    # Base to Global
    R_base2global = R_global2base.T
    #d_base2global = -R_base2global @ d_global2base

    T_base2global = np.eye(4)
    T_base2global[0:3, 0:3] = R_base2global
    T_base2global[0:3, 3:4] = d_global2base

    # TCP to T : active interpretation of the transformation, moves the point from TCP to T inside the same "world" frame
    R_tcp2t = np.array([
        [1, 0, 0],
        [0, np.cos(theta_h - np.pi / 2), -np.sin(theta_h - np.pi / 2)],
        [0, np.sin(theta_h - np.pi / 2),  np.cos(theta_h - np.pi / 2)]
    ])
    d_tcp2t = np.array([[0], [-Rn * np.sin(theta_h)], [Rn * (np.cos(theta_h) - 1)]])

    T_tcp2t = np.eye(4)
    T_tcp2t[0:3, 0:3] = R_tcp2t
    T_tcp2t[0:3, 3:4] = d_tcp2t

    # T to TCP : active interpretation of the transformation, moves the point from T to TCP inside the same "base" frame
    R_t2tcp = R_tcp2t.T
    d_t2tcp = -R_tcp2t @ d_tcp2t

    T_t2tcp = np.eye(4)
    T_t2tcp[0:3, 0:3] = R_t2tcp
    T_t2tcp[0:3, 3:4] = d_t2tcp

    # Constant Rotation Matrix (X-axis -90 deg)
    R_x_m90 = np.array([
        [1, 0, 0],
        [0, np.cos(-np.pi / 2), -np.sin(-np.pi / 2)],
        [0, np.sin(-np.pi / 2),  np.cos(-np.pi / 2)]
    ])

    # -------------------------------------------------------------------------
    # 3. Angle discretization & Buffer Init
    # -------------------------------------------------------------------------
    phi_deg = np.linspace(180, 360, N) # np.arange(180, 361, 10)  # Generates 19 elements matching 180:10:360
    phi = phi_deg * np.pi / 180
    delta_z = 0.0

    # Preallocate result matrices (3 rows, 19 columns)
    P_t = np.zeros((3, N))
    O_t = np.zeros((3, N))

    O_t2tcp = np.zeros((3, N))

    P_tcp = np.zeros((3, N))
    O_tcp = np.zeros((3, N))

    P_base = np.zeros((3, N))
    O_base = np.zeros((3, N))
    P_t_base = np.zeros((3, N))
    O_t_base = np.zeros((3, N))

    # -------------------------------------------------------------------------
    # 4. Main Computation Loop
    # -------------------------------------------------------------------------
    for i in range(N):
        # Radius computation
        P_r_val = np.sqrt(b**2 / (1 - k * (np.cos(phi[i]))**2))
        
        # Target position
        P_t_col = np.array([[0.0], [P_r_val * np.cos(phi[i])], [P_r_val * np.sin(phi[i])]])
        P_t[:, i : i + 1] = P_t_col + elipse_center_global + np.array([[0], [0], [delta_z]])

        # Tangent vector derivative logic
        P_r_dot_val = (-0.5 * b * np.sin(2 * phi[i]) * k) / (1 - k * (np.cos(phi[i]))**2)**1.5
        
        O_t_col = np.array([
            [0],
            [-P_r_val * np.sin(phi[i]) + P_r_dot_val * np.cos(phi[i])],
            [+P_r_val * np.cos(phi[i]) + P_r_dot_val * np.sin(phi[i])]
        ])
        O_t[:, i : i + 1] = (O_t_col / np.linalg.norm(O_t_col)) 
        
        # Homogeneous transformations to TCP space
        P_t_homogen = np.append(P_t[:, i], 1.0).reshape(4, 1)
        P_tcp_homogen = T_t2tcp @ P_t_homogen
        P_tcp[:, i] = P_tcp_homogen[0:3].flatten()
        O_tcp[:, i] = (R_x_m90 @ R_t2tcp @ O_t[:, i : i + 1]).flatten()
        O_t2tcp[:, i] = (R_t2tcp @ O_t[:, i : i + 1]).flatten()

        # Homogeneous transformations to Robot Base space
        P_base_homogen = T_global2base @ P_tcp_homogen
        P_base[:, i] = P_base_homogen[0:3].flatten()
        O_base[:, i] = (R_global2base @ O_tcp[:, i : i + 1]).flatten()

        P_t_base_homogen = T_global2base @ P_t_homogen
        P_t_base[:, i] = P_t_base_homogen[0:3].flatten()
        O_t_base[:, i] = (R_global2base @ O_t[:, i : i + 1]).flatten()
        
        print("ecllipse_center_base: ", T_global2base @ np.append(elipse_center_global, 1.0).reshape(4, 1))
        # Matching MATLAB index criteria (1-based > 10 maps to 0-based >= 10)
        if i >= 10:
            delta_z += 0 * 0.2

    T = O_t2tcp#np.array([O_t2tcp[0, :], O_t2tcp[1, :], O_t2tcp[2, :]]) - np.array([P_tcp[0, :], P_tcp[1, :], P_tcp[2, :]])
    BiNor = O_tcp# np.array([O_tcp[0, :], O_tcp[1, :], O_tcp[2, :]]) - np.array([P_tcp[0, :], P_tcp[1, :], P_tcp[2, :]])# np.cross(T, N, axis=0)
    N = np.cross([T[0, :], T[1, :], T[2, :]], [BiNor[0, :], BiNor[1, :], BiNor[2, :]], axis=0)
    N = N / np.linalg.norm(N, axis=0)  # Normalize the normal vector


    T_base = O_t_base
    BiNor_base = O_base
    N_base = np.cross([T_base[0, :], T_base[1, :], T_base[2, :]], [BiNor_base[0, :], BiNor_base[1, :], BiNor_base[2, :]], axis=0)
    N_base = N_base / np.linalg.norm(N_base, axis=0)  # Normalize the normal vector
    # -------------------------------------------------------------------------
    # 5. Plotting Generation (3 Subplots matching MATLAB figure 1)
    # -------------------------------------------------------------------------
    if show_plot:
        fig = plt.figure(figsize=(18, 6))

        # Subplot 1: Target Needle Tip
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        ax1.plot3D(P_t[0, :], P_t[1, :], P_t[2, :], 'b', linewidth=2)
        ax1.quiver(P_t[0, :], P_t[1, :], P_t[2, :], O_t[0, :], O_t[1, :], O_t[2, :], color='r', length=0.2)
        ax1.set_title('Target Needle Tip\n(global pos + global orient)')
        ax1.set_xlabel('x_t')
        ax1.set_ylabel('y_t')
        ax1.set_zlabel('z_t')
        #ax1.view_init(elev=90, azim=0) # Emulates view(90,0)
        ax1.grid(True)

        # Subplot 2: Target TCP
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        ax2.plot3D(P_tcp[0, :], P_tcp[1, :], P_tcp[2, :], 'b', linewidth=2)
        ax2.quiver(P_tcp[0, :], P_tcp[1, :], P_tcp[2, :], O_tcp[0, :], O_tcp[1, :], O_tcp[2, :], color='r', length=0.2)
        ax2.set_title('Target tcp\n(global position)')
        ax2.set_xlabel('x_tcp')
        ax2.set_ylabel('y_tcp')
        ax2.set_zlabel('z_tcp')
        ax2.grid(True)

        # Subplot 3: Required Base Position
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        ax3.plot3D(P_base[0, :], P_base[1, :], P_base[2, :], 'b', linewidth=2)
        ax3.quiver(P_base[0, :], P_base[1, :], P_base[2, :], O_base[0, :], O_base[1, :], O_base[2, :], color='r', length=0.2)
        ax3.quiver(P_base[0, :], P_base[1, :], P_base[2, :], O_t_base[0, :], O_t_base[1, :], O_t_base[2, :], color='g', length=0.2)
        ax3.quiver(P_base[0, :], P_base[1, :], P_base[2, :], N_base[0, :], N_base[1, :], N_base[2, :], color='y', length=0.2)
        ax3.set_title('Required Base\nposition and orientation')
        ax3.set_xlabel('x_base')
        ax3.set_ylabel('y_base')
        ax3.set_zlabel('z_base')
        ax3.grid(True)

        plt.tight_layout()
        plt.ion()
        plt.show()

    return tuple(P_tcp, O_t2tcp, O_tcp, N), tuple(P_base, O_t_base, O_base, N_base)
# %%
