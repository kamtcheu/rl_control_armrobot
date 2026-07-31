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
site_id = model.site("tcp_site").id
tool_id = model.site("attachment_site").id
wrist_3_id = model.body("wrist_3_link").id
wrist_joint_id = model.joint("wrist_3_joint").id
base_id = model.body("base").id
trail = []
MAX_POINTS = 500
time.sleep(15)  
ellip,_, _, needl, _ = elliptical_trajectory(N=19, show_plot=False)
#ellip = ellip*1e-3
print(ellip)
ellip2 = [np.array([ellip[0,i], ellip[1,i], ellip[2,i]]) for i in range(ellip.shape[1])]
needl2 = [np.array([needl[0,i], needl[1,i], needl[2,i]]) for i in range(needl.shape[1])]
print(ellip2)
init_joints_degrees = [0]*6#[-10.042, -93.752, -63.163, -109.568, 76.181, -11.005]
init_joints_grad = [angle * np.pi / 180 for angle in init_joints_degrees]
#init_joints_grad[0] = -np.pi/2
        
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        if model.nu > 0:
            data.ctrl[:] = np.array(init_joints_grad)#[0, -1.51, 1.51, -1.51, -1.51, 0])#1 * np.random.randn(model.nu)
        mujoco.mj_step(model, data)
        # print("qpos:", data.qpos)
        # print("qvel:", data.qvel)
        # print("ctrl:", data.ctrl)
        print("site xpos:", data.site_xpos[site_id])
        
        print("robot nativ tcp xpos:", data.site_xpos[tool_id])
        print("site pos base:\n", model.site_pos[site_id])
        #scipy is scalar last while mujoco is scalar first, so we need to reorder the quaternion components
        sit_quat = [float(x) for x in R.from_matrix(data.site_xmat[site_id].reshape(3, 3)).as_quat()]
        # print("site xquat:\n", [sit_quat[3], sit_quat[0], sit_quat[1], sit_quat[2]])
        # print("wrist_3 xquat:\n", data.xquat[wrist_3_id])
        print("site xmat:\n", data.site_xmat[site_id].reshape(3, 3))
        print("native tcp site xmat:\n", data.site_xmat[tool_id].reshape(3, 3))
        print("wrist_3 xmat:\n", data.xmat[wrist_3_id].reshape(3, 3))
        print("base coordinates:\n", data.body(base_id).xpos)
        #print("wrist_3 joint xquat:", data.xquat[wrist_joint_id])
        p = data.site_xpos[site_id].copy()
        #print(p)
        trail = ellip2#[:, [1, 2, 0]] #trail.append(p)
        #trail = needl2#[:, [1, 2, 0]] #trail.append(p)
        # if len(trail) > MAX_POINTS:
        #     trail.pop(0)
        #print(ellip)
        with viewer.lock():
            viewer.user_scn.ngeom = 0
            i = 0
            vor_last_pt = None
            # draw points
            trail.append(p)
            for pt in trail:
                # print(pt)
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[i],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([0.001, 0, 0]),
                    pos=pt,
                    mat=np.eye(3).reshape(-1),
                    rgba=np.array([1, 0, 0, 1]),
                )
                if i > 0 :
                    point = viewer.user_scn.geoms[i-1]
                    mujoco.mjv_connector(point ,mujoco.mjtGeom.mjGEOM_LINE , 2, vor_last_pt, pt)
                
                vor_last_pt = pt
                i += 1
                viewer.user_scn.ngeom += 1
            # i=1
            # vor_last_pt = None
            # for pt in trail2:
            #     # print(pt)
            #     mujoco.mjv_initGeom(
            #         viewer.user_scn.geoms[viewer.user_scn.ngeom + i],
            #         type=mujoco.mjtGeom.mjGEOM_SPHERE,
            #         size=np.array([0.001, 0, 0]),
            #         pos=pt,
            #         mat=np.eye(3).reshape(-1),
            #         rgba=np.array([1, 1, 0, 1]),
            #     )
            #     if i > 1 :
            #         point = viewer.user_scn.geoms[viewer.user_scn.ngeom + i-1]
            #         mujoco.mjv_connector(point ,mujoco.mjtGeom.mjGEOM_LINE , 2, vor_last_pt, pt)
                
            #     vor_last_pt = pt
            #     i += 1

            #     viewer.user_scn.ngeom += 1

        viewer.sync()
        time.sleep(model.opt.timestep)