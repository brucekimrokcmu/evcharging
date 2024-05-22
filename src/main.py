import os
import time
import numpy as np
import mujoco
import mujoco.viewer as viewer
from dm_control.utils import inverse_kinematics as ik


# Set up the simulation parameters
SIM_DURATION = 5
MODEL_PATH = '../data/universal_robots_ur10e/scene.xml'
INTEGRATION_DT: float = 1.0
DAMPING: float = 1e-4
GRAVITY_COMPENSATION: bool = True
DT: float = 0.002
MAX_ANGVEL = 0.0


def get_qpos(model, data):
    qpos_array = np.zeros(model.nq, dtype=np.float64)
    mujoco.mj_getState(model, data, qpos_array, mujoco.mjtState.mjSTATE_QPOS)
    
    return qpos_array


def get_xpos_xquat(model, data):
    mujoco.mj_forward(model, data)
    xpos = np.copy(np.reshape(data.xpos, (model.nbody, 3)))
    xquat = np.copy(np.reshape(data.xquat, (model.nbody, 4)))
    return xpos, xquat

def get_random_target_pose(model, data):
    target_position = np.random.uniform(-0.5, 0.5, 3)
    target_quat = np.random.random(4)
    target_quat /= np.linalg.norm(target_quat)
    return target_position, target_quat

def main(): 
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    model = mujoco.MjModel.from_xml_path(os.path.join(curr_dir, MODEL_PATH)) 
    if not model:
        mujoco.mju_error("Failed to load model")
    
    data = mujoco.MjData(model)
    if not data:
        mujoco.mju_error("Failed to create data object")

    model.opt.timestep = DT

    site_id = model.site("attachment_site").id
    if site_id == -1:
        raise ValueError("Site not found")

    body_names = [
        "shoulder_link",
        "upper_arm_link",
        "forearm_link",
        "wrist_1_link",
        "wrist_2_link",
        "wrist_3_link"
    ]

    body_ids = [model.body(name).id for name in body_names]
    if GRAVITY_COMPENSATION:
        model.body_gravcomp[body_ids] = 1.0
    
    joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint"
    ]    

    dof_ids = np.array([model.joint(name).id for name in joint_names])

    actuator_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3"
    ]    

    actuator_ids = np.array([model.actuator(name).id for name in actuator_names])

    key_id = model.key("home").id

    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
    #     viewer.cam.distance *= 2.5
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        start = time.time()
        while viewer.is_running() and time.time() - start < SIM_DURATION:
            step_start = time.time()            

            target_position, target_quat = get_random_target_pose(model, data)
            qpos = get_qpos(model, data)
            xpos, xquat = get_xpos_xquat(model, data)

            #TODO: Implement inverse kinematics
            ik_result = ik.qpos_from_site_pose(
                physics=model,
                site_name='attachment_site',
                target_pos=target_position,
                target_quat=target_quat,
                joint_ids=dof_ids,
                initial_joint_angles=qpos
            )
            if ik_result.success:
                q = ik_result.qpos
                dq = ik_result.qvel  

                mujoco.mj_integratePos(model, q, dq, INTEGRATION_DT)

                np.clip(q, *model.jnt_range.T, out=q)
                data.ctrl[actuator_ids] = q[dof_ids]


            # Step the simulation
            mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next_step = DT - (time.time() - step_start)  
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()