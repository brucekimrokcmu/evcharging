import os
import time
# import unittest

import mujoco 
from mujoco import viewer
from IPython.display import clear_output
import numpy as np

SIM_DURATION = 10
MODEL_PATH = '../data/universal_robots_ur10e/scene.xml'

def get_joint_values(data):
    print(data.qpos)
    return data.qpos

def get_pose(data):
    print(data.xpos)
    return data.xpos

"""
Switch over to dm control?
 https://github.com/google-deepmind/dm_control/blob/main/dm_control/utils/inverse_kinematics.py
"""

def numerical_ik_solver(model, data, target_position, max_iterations=100, tolerance=1e-4):
    initial_joint_values = data.qpos.copy()

    def forward_kinematics(joint_values):
        data.qpos = joint_values
        mujoco.mj_kinematics(model, data)
        return get_pose(data)

    joint_values = initial_joint_values

    for _ in range(max_iterations):
        current_position = forward_kinematics(joint_values)
        error = target_position - current_position
        if np.linalg.norm(error) < tolerance:
            break
        
        jacobian = mujoco.mj_jac(model, data)

        joint_updates = np.linalg.pinv(jacobian) @ error
        joint_values += joint_updates
    
    data.qpos = joint_values
    return joint_values

def main():
    # Load the model
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    model = mujoco.MjModel.from_xml_path(os.path.join(curr_dir, MODEL_PATH)) 
    data = mujoco.MjData(model)

    initial_joint_values = get_joint_values(data).copy()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance *= 2.5
        start = time.time()
        while viewer.is_running() and time.time() - start < SIM_DURATION:
            step_start = time.time()
            mujoco.mj_step(model, data)

            elapsed_time = time.time() - start
            x = get_joint_values(data)
            data.qpos = initial_joint_values + (-2 * elapsed_time / SIM_DURATION)

            # with viewer.lock():
                # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time %2) # Toggle contact points
            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)  
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()