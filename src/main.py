import os
import time
import numpy as np
from dm_control import mujoco
from dm_control.mujoco.wrapper.mjbindings import mjlib, enums
from dm_control import mjcf
from dm_control import suite

from dm_control.utils import inverse_kinematics as ik
from dm_control import _render
import mujoco.viewer as viewer

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
    initial_xpos = physics.bind(site).xpos
    diff_xpos = target_pos - initial_xpos

    if num_of_interpolation < 2:
        raise ValueError("num_of_interpolation must be greater than 2")
    
    physics.data.qpos[:] = physics.data.qpos.copy()

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
    ur10e_model = os.path.join(curr_dir, '../data/universal_robots_ur10e/ur10e.xml')
    physics = mujoco.Physics.from_xml_path(ur10e_model)

    target_pos, target_quat = get_random_target_pose()

    ik_result = ik.qpos_from_site_pose(physics, 'attachment_site', target_pos, target_quat)
    print(ik_result)

    # TODO: render the simulation
    for _ in range(int(SIM_DURATION * 60)):
        physics.step()
        physics.render()

if __name__ == "__main__":
    main()