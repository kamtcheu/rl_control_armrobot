import time
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
import os
from trajectory import elliptical_trajectory
model = mujoco.MjModel.from_xml_path('./universal_robots_ur3e-main/scene.xml')
#model.nu = 0.1
data = mujoco.MjData(model)
print(model.nu)
site_id = model.site("needle_holder_site").id
tool_id = model.site("attachment_site").id
wrist_3_id = model.body("wrist_3_link").id
wrist_joint_id = model.joint("wrist_3_joint").id
base_id = model.body("base").id
trail = []
MAX_POINTS = 500
time.sleep(15)  
ellip, needl = elliptical_trajectory(N=19, show_plot=False)
ellip = ellip[0]*1e-3
needl = needl[0]*1e-3
#ellip = ellip*1e-3
print(ellip)
ellip2 = [np.array([ellip[0,i], ellip[1,i], ellip[2,i]]) for i in range(ellip.shape[1])]
needl2 = [np.array([needl[0,i], needl[1,i], needl[2,i]]) for i in range(needl.shape[1])]
print(ellip2)
init_joints_degrees = np.array([-21.58, -97.97, -66.38, -106.18, 88.76, -21.74])#[0]*6#[-10.042, -93.752, -63.163, -109.568, 76.181, -11.005]
init_joints_grad = [angle * np.pi / 180 for angle in init_joints_degrees]
#init_joints_grad[0] = -np.pi/2
joints = [np.array([-0.3775, -1.7001, -1.1659, -1.8421,  1.5596, -0.667 ]), np.array([-0.3838, -1.703 , -1.1644, -1.8454,  1.5624, -1.2208]), np.array([-0.3869, -1.7074, -1.1613, -1.8407,  1.5634, -1.5438]), np.array([-0.3877, -1.7108, -1.1581, -1.842 ,  1.5641, -1.7762]), np.array([-0.3901, -1.7132, -1.1554, -1.8412,  1.5647, -1.9244]), np.array([-0.389 , -1.7144, -1.1541, -1.8402,  1.5648, -2.0295]), np.array([-0.392 , -1.7167, -1.1524, -1.8406,  1.5661, -2.1322]), np.array([-0.3836, -1.7174, -1.1516, -1.8306,  1.5626, -2.1978]), np.array([-0.3843, -1.717 , -1.1511, -1.8314,  1.5644, -2.2615]), np.array([-0.3853, -1.7167, -1.1504, -1.833 ,  1.5657, -2.3243]), np.array([-0.3853, -1.7164, -1.1497, -1.8331,  1.5663, -2.3869]), np.array([-0.386 , -1.718 , -1.1489, -1.8337,  1.567 , -2.4495]), np.array([-0.3838, -1.7195, -1.1484, -1.8321,  1.5672, -2.5135]), np.array([-0.3821, -1.7183, -1.1489, -1.8271,  1.5675, -2.5786]), np.array([-0.3849, -1.722 , -1.1468, -1.836 ,  1.5707, -2.6787]), np.array([-0.3868, -1.7213, -1.1466, -1.8348,  1.5716, -2.7885]), np.array([-0.3868, -1.7197, -1.1484, -1.8339,  1.5732, -2.9373]), np.array([-0.3892, -1.7243, -1.1463, -1.8325,  1.5764, -3.1663]), np.array([-0.3915, -1.7218, -1.1485, -1.8287,  1.5794, -3.485 ]), np.array([-0.3972, -1.7224, -1.1479, -1.8284,  1.585 , -3.9264])]#[:, [1, 2, 0]] #trail.append(p)
steps_per_action = 20
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat = np.array([0.02, 0.03, 0.19])
    viewer.cam.distance = 0.45
    viewer.cam.azimuth = 4
    viewer.cam.elevation = -18
    for q in joints:
        data.qpos[:6] = q
        #data.ctrl[:6] = q
        mujoco.mj_forward(model, data)
        #for _ in range(steps_per_action):
        mujoco.mj_step(model, data)
        viewer.sync()

        time.sleep(0.5)
        #viewer.sync()
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.01)
        #time.sleep(0.5)
    # while viewer.is_running():
    #     if model.nu > 0:
    #         data.ctrl[:] = np.array(init_joints_grad)#[0, -1.51, 1.51, -1.51, -1.51, 0])#1 * np.random.randn(model.nu)
    #     mujoco.mj_step(model, data)
    #     # print("qpos:", data.qpos)
    #     # print("qvel:", data.qvel)
    #     # print("ctrl:", data.ctrl)
    #     print("site xpos:", data.site_xpos[site_id])
        
    #     print("robot nativ tcp xpos:", data.site_xpos[tool_id])
    #     print("site pos base:\n", model.site_pos[site_id])
    #     #scipy is scalar last while mujoco is scalar first, so we need to reorder the quaternion components
    #     sit_quat = [float(x) for x in R.from_matrix(data.site_xmat[site_id].reshape(3, 3)).as_quat()]
    #     # print("site xquat:\n", [sit_quat[3], sit_quat[0], sit_quat[1], sit_quat[2]])
    #     # print("wrist_3 xquat:\n", data.xquat[wrist_3_id])
    #     print("site xmat:\n", data.site_xmat[site_id].reshape(3, 3))
    #     print("native tcp site xmat:\n", data.site_xmat[tool_id].reshape(3, 3))
    #     print("wrist_3 xmat:\n", data.xmat[wrist_3_id].reshape(3, 3))
    #     print("base coordinates:\n", data.body(base_id).xpos)
    #     #print("wrist_3 joint xquat:", data.xquat[wrist_joint_id])
    #     p = data.site_xpos[site_id].copy()
    #     #print(p)
    #     trail = ellip2#[:, [1, 2, 0]] #trail.append(p)
    #     #trail = needl2#[:, [1, 2, 0]] #trail.append(p)
    #     # if len(trail) > MAX_POINTS:
    #     #     trail.pop(0)
    #     #print(ellip)
    #     with viewer.lock():
    #         viewer.user_scn.ngeom = 0
    #         i = 0
    #         vor_last_pt = None
    #         # draw points
    #         trail.append(p)
    #         for pt in trail:
    #             # print(pt)
    #             mujoco.mjv_initGeom(
    #                 viewer.user_scn.geoms[i],
    #                 type=mujoco.mjtGeom.mjGEOM_SPHERE,
    #                 size=np.array([0.001, 0, 0]),
    #                 pos=pt,
    #                 mat=np.eye(3).reshape(-1),
    #                 rgba=np.array([1, 0, 0, 1]),
    #             )
    #             if i > 0 :
    #                 point = viewer.user_scn.geoms[i-1]
    #                 mujoco.mjv_connector(point ,mujoco.mjtGeom.mjGEOM_LINE , 2, vor_last_pt, pt)
                
    #             vor_last_pt = pt
    #             i += 1
    #             viewer.user_scn.ngeom += 1
    #         # i=1
    #         # vor_last_pt = None
    #         # for pt in trail2:
    #         #     # print(pt)
    #         #     mujoco.mjv_initGeom(
    #         #         viewer.user_scn.geoms[viewer.user_scn.ngeom + i],
    #         #         type=mujoco.mjtGeom.mjGEOM_SPHERE,
    #         #         size=np.array([0.001, 0, 0]),
    #         #         pos=pt,
    #         #         mat=np.eye(3).reshape(-1),
    #         #         rgba=np.array([1, 1, 0, 1]),
    #         #     )
    #         #     if i > 1 :
    #         #         point = viewer.user_scn.geoms[viewer.user_scn.ngeom + i-1]
    #         #         mujoco.mjv_connector(point ,mujoco.mjtGeom.mjGEOM_LINE , 2, vor_last_pt, pt)
                
    #         #     vor_last_pt = pt
    #         #     i += 1

    #         #     viewer.user_scn.ngeom += 1

    #     viewer.sync()
    #     time.sleep(model.opt.timestep)