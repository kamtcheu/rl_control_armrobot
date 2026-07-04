import time
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
import os
model = mujoco.MjModel.from_xml_path('./universal_robots_ur3e-main/scene.xml')
#model.nu = 0.1
data = mujoco.MjData(model)
print(model.nu)
site_id = model.site("attachment_site").id
wrist_3_id = model.body("wrist_3_link").id
wrist_joint_id = model.joint("wrist_3_joint").id
trail = []
MAX_POINTS = 500
time.sleep(15)  
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        if model.nu > 0:
            data.ctrl[:] = np.array([0, -1.51, 1.51, -1.51, -1.51, 0])#1 * np.random.randn(model.nu)
        mujoco.mj_step(model, data)
        print("qpos:", data.qpos)
        print("qvel:", data.qvel)
        print("ctrl:", data.ctrl)
        print("site xpos:", data.site(site_id).xpos)
        #scipy is scalar last while mujoco is scalar first, so we need to reorder the quaternion components
        sit_quat = [float(x) for x in R.from_matrix(data.site_xmat[site_id].reshape(3, 3)).as_quat()]
        # print("site xquat:\n", [sit_quat[3], sit_quat[0], sit_quat[1], sit_quat[2]])
        # print("wrist_3 xquat:\n", data.xquat[wrist_3_id])
        print("site xmat:\n", data.site_xmat[site_id].reshape(3, 3))
        print("wrist_3 xmat:\n", data.xmat[wrist_3_id].reshape(3, 3))
        #print("wrist_3 joint xquat:", data.xquat[wrist_joint_id])
        p = data.site(site_id).xpos.copy()
        #print(p)
        trail.append(p)
        if len(trail) > MAX_POINTS:
            trail.pop(0)

        with viewer.lock():
            viewer.user_scn.ngeom = 0
            i = 0
            vor_last_pt = None
            # draw points
            for pt in trail:
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

        viewer.sync()
        time.sleep(model.opt.timestep)