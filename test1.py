import time
import numpy as np
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path('./universal_robots_ur3e-main/ur3e.xml')
#model.nu = 0.1
data = mujoco.MjData(model)
print(model.nu)
site_id = model.site("attachment_site").id
trail = []
MAX_POINTS = 500

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        if model.nu > 0:
            data.ctrl[:] = 3 * np.random.randn(model.nu)
        mujoco.mj_step(model, data)
        print(data.ctrl)
        p = data.site(site_id).xpos.copy()
        print(p)
        trail.append(p)
        if len(trail) > MAX_POINTS:
            trail.pop(0)

        with viewer.lock():
            viewer.user_scn.ngeom = 0
            i = 0

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
                i += 1

            viewer.user_scn.ngeom = i

        viewer.sync()
        time.sleep(model.opt.timestep)