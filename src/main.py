import os
import time
import numpy as np
from dm_control import mujoco as dm_mujoco
from dm_control.mujoco.wrapper.mjbindings import mjlib, enums
from dm_control.utils import inverse_kinematics as ik
import mujoco.viewer 

# Set up the simulation parameters
SIM_DURATION = 5.0
# MODEL_PATH = 

def get_random_target_pose():
    target_pos = np.random.uniform(-0.5, 0.5, 3)
    target_quat = np.random.random(4)
    target_quat /= np.linalg.norm(target_quat)
    return target_pos, target_quat

# TODO: implement set_site_to_xpos
def linear_interpolate_planner(physics, site, target_pos, target_quat=None, num_of_interpolation=20, max_ik_attempts=10):
    if target_quat is None:
        target_quat = np.array([1, 0, 0, 0])

    initial_xpos = physics.data.xpos
    diff_xpos = target_pos - initial_xpos

    if num_of_interpolation < 2:
        raise ValueError("num_of_interpolation must be greater than 2")
    
    for step in range(num_of_interpolation):
        interp_xpos = initial_xpos + diff_xpos * (step / (num_of_interpolation - 1))

        for attempt in range(max_ik_attempts):
            ik_result = ik.qpos_from_site_pose(physics, site, interp_xpos, target_quat)
            if ik_result.success:
                physics.data.qpos[:] = ik_result.qpos
                mjlib.mj_forward(physics.model, physics.data)
                break
            else:
                print(f"Failed to find a solution at step {step} attempt {attempt}")

def main(): 
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ur10e_model = os.path.join(curr_dir, '../data/universal_robots_ur10e/scene.xml')
    physics = dm_mujoco.Physics.from_xml_path(ur10e_model)
    model = physics.model.ptr
    data = physics.data.ptr

    initial_qpos = np.array([1.57, -0.78, -0.5, 0.5, 0.5, 0.5])
    physics.data.qpos[:] = initial_qpos
    physics.step()

    with mujoco.viewer.launch_passive(model=model, data=data,
                                      show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.distance *= 2.5

        target_pos, target_quat = get_random_target_pose()
        
        print(physics.data.qpos)
        # TODO figure out how to achieve target_quat
        ik_result = ik.qpos_from_site_pose(physics, 'attachment_site', target_pos, target_quat=None)
        print(ik_result)    
        while viewer.is_running():

            # linear_interpolate_planner(physics, 'attachment_site', target_pos, target_quat)
            viewer.sync()

if __name__ == "__main__":
    main()