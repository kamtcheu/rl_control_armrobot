import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation, Slerp


# ============================================================
# Helper: plot one rotation matrix as 3 orthogonal basis vectors
# ============================================================

def plot_rotation_frame(ax, R, origin, label, alpha=1.0):
    """
    R      : 3x3 rotation matrix
    origin : [x, y, z] location where the frame is drawn
    """

    origin = np.asarray(origin)

    # Columns of R are the rotated basis vectors
    x_axis = R[:, 0]
    y_axis = R[:, 1]
    z_axis = R[:, 2]

    ax.quiver(
        origin[0], origin[1], origin[2],
        x_axis[0], x_axis[1], x_axis[2],
        length=0.8,
        normalize=True,
        color = 'r',
        alpha=alpha
    )

    ax.quiver(
        origin[0], origin[1], origin[2],
        y_axis[0], y_axis[1], y_axis[2],
        length=0.8,
        normalize=True,
        color = 'g',
        alpha=alpha
    )

    ax.quiver(
        origin[0], origin[1], origin[2],
        z_axis[0], z_axis[1], z_axis[2],
        length=0.8,
        normalize=True,
        color = 'b',
        alpha=alpha
    )

    ax.text(
        origin[0],
        origin[1],
        origin[2] + 0.1,
        label
    )


# ============================================================
# Define the two orientations
# ============================================================

# Start orientation
R0 = np.eye(3)

# Example target orientation:
# rotation of 100 degrees around axis [1, 1, 0.5]
axis = np.array([1.0, 0.0, 0.0])
axis = axis / np.linalg.norm(axis)

angle_deg = 90.0
angle_rad = np.deg2rad(angle_deg)

R1 = Rotation.from_rotvec(
    axis * angle_rad
).as_matrix()


# ============================================================
# Interpolation parameter
# ============================================================

s = 0.6

# s = 0   -> R0
# s = 1   -> R1


# ============================================================
# SLERP
# ============================================================

key_rotations = Rotation.from_matrix(
    np.stack([R0, R1])
)

slerp = Slerp(
    [0.0, 1.0],
    key_rotations
)

R_interp = slerp([s]).as_matrix()[0]


# ============================================================
# Print matrices
# ============================================================

print("R0:")
print(R0)

print("\nR interpolated at s =", s)
print(R_interp)

print("\nR1:")
print(R1)


# ============================================================
# Plot
# ============================================================

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")


# Draw the three frames at different positions only
# for visualization purposes
plot_rotation_frame(
    ax,
    R0,
    origin=[-1.5, 0, 0],
    label="R0",
    alpha=0.4
)

plot_rotation_frame(
    ax,
    R_interp,
    origin=[0, 0, 0],
    label=f"R(s={s:.2f})",
    alpha=1.0
)

plot_rotation_frame(
    ax,
    R1,
    origin=[1.5, 0, 0],
    label="R1",
    alpha=0.7
)


# ============================================================
# Plot formatting
# ============================================================

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1.5, 1.5)

ax.set_box_aspect([5, 3, 3])

ax.set_title(
    f"SLERP interpolation between two rotation matrices, s = {s:.2f}"
)

plt.show(block=True)