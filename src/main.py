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

def get_random_target_pose(qpos_max, qpos_min, quat):
    target_pos = np.random.uniform(-0.5, 0.5, 3)
    target_quat = np.random.random(4)
    target_quat /= np.linalg.norm(target_quat)
    return target_pos, target_quat

# TODO: implement set_site_to_xpos
def generate_trajectory(start_pos, start_quat, target_pos, target_quat, num_waypoints=1000):
    trajectory = []

    return trajectory

def test_ik(physics, target_pos=None, target_quat=None): 
    
    qpos = physics.data.qpos
    xpos = physics.data.xpos
    site_xpos = physics.data.site_xpos[-1,:]
    print("Initial qpos: ", qpos)
    print("Initial xpos: ", site_xpos)
    if target_pos is None:
        target_pos = site_xpos
    
    ik_result = ik.qpos_from_site_pose(physics, 'attachment_site', target_pos, target_quat=None)
    print("IK result: ", ik_result)    
    return ik_result

def main(): 
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ur10e_model = os.path.join(curr_dir, '../data/universal_robots_ur10e/scene.xml')
    physics = dm_mujoco.Physics.from_xml_path(ur10e_model)
    model = physics.model.ptr
    data = physics.data.ptr

    initial_qpos = np.array([1.57, -2.0, 1.3, 0.5, 0.5, 0.5])
    physics.data.qpos[:] = initial_qpos

    test_ik(physics)


    # with mujoco.viewer.launch_passive(model=model, data=data,
    #                                   show_left_ui=False, show_right_ui=False) as viewer:
    #     viewer.cam.distance *= 2.5
    #     target_pos, target_quat = get_random_target_pose()

    #     while viewer.is_running():

    #         viewer.sync()

if __name__ == "__main__":
    main()