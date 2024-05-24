import os
import time
import numpy as np
from dm_control import mujoco
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings import mjlib

from dm_control import mjcf
from dm_control import suite

from dm_control.utils import inverse_kinematics as ik
import mujoco.viewer as viewer

# Set up the simulation parameters
SIM_DURATION = 5.0
# MODEL_PATH = 

def get_random_target_pose():
    target_pos = np.random.uniform(-0.5, 0.5, 3)
    target_quat = np.random.random(4)
    target_quat /= np.linalg.norm(target_quat)
    return target_pos, target_quat

def main(): 
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ur10e_model = os.path.join(curr_dir, '../data/universal_robots_ur10e/ur10e.xml')

    physics = mujoco.Physics.from_xml_path(ur10e_model)
    qpos = physics.get_state()

    target_pos, target_quat = get_random_target_pose()
    ik_result = ik.qpos_from_site_pose(physics, 'attachment_site', target_pos, target_quat)
    print(ik_result)


if __name__ == "__main__":
    main()