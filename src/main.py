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

# TODO slerp utility function for quaternion interpolation
def slerp(q0, q1, t):
    pass

def linear_trajectory_in_workspace(start_pos, end_pos, num_waypoints=100):
    return np.linspace(start_pos, end_pos, num=num_waypoints)

def solve_ik_for_waypoints(physics, waypoints, site_name='attachment_site'):
    joint_poses = []
    for waypoint in waypoints:
        result = ik.qpos_from_site_pose(
            physics,
            site_name,
            waypoint,
            target_quat=None)  
        if result.success:
            joint_poses.append(result.qpos)
        else:
            print(f"IK failed at waypoint {waypoint}")
            return None
    return joint_poses

def cubic_polynomial_interpolation(q0, qf, v0=None, vf=None, num_points=100, duration=2.0):
    if v0 is None:
        v0 = np.zeros_like(q0)
    if vf is None:
        vf = np.zeros_like(qf)
    
    t = np.linspace(0, duration, num=num_points)
    
    A = np.array([
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [duration**3, duration**2, duration, 1],
        [3*duration**2, 2*duration, 1, 0]
    ])
    
    b = np.column_stack([q0, v0, qf, vf])
    c = np.linalg.solve(A, b)
    
    q_des = np.einsum('i,ij->ij', t**3, c[0]) + \
            np.einsum('i,ij->ij', t**2, c[1]) + \
            np.einsum('i,ij->ij', t, c[2]) + c[3]
    
    return q_des

def generate_trajectory(physics, start_pos, end_pos, num_waypoints=100, duration=2.0, site_name='attachment_site'):
    workspace_trajectory = linear_trajectory_in_workspace(start_pos, end_pos, num_waypoints)
    
    joint_poses = solve_ik_for_waypoints(physics, [start_pos, end_pos], site_name)
    if joint_poses is None:
        print("IK failed for start or end pose.")
        return None
    
    q0, qf = joint_poses[0], joint_poses[-1]
    v0, vf = np.zeros_like(q0), np.zeros_like(qf)    
    trajectory = cubic_polynomial_interpolation(q0, qf, v0, vf, num_points=num_waypoints, duration=duration)
    
    return trajectory, workspace_trajectory


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