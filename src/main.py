import os
import time
import numpy as np
from dm_control import mujoco as dm_mujoco
from dm_control.mujoco.wrapper.mjbindings import mjlib, enums
from dm_control.utils import inverse_kinematics as ik
import mujoco.viewer 

# Set up the simulation parameters
SIM_DURATION = 5.0

def get_random_pose(qpos_max=0.5, qpos_min=-0.5):
    random_pos = np.random.uniform(qpos_min, qpos_max, 3)
    random_quaternion = np.random.normal(0, 1, 4)
    random_quaternion /= np.linalg.norm(random_quaternion)
    random_pose = np.concatenate([random_pos, random_quaternion])
    return random_pose


# TODO slerp utility function for quaternion interpolation
def slerp(q0, q1, t):
    pass

# TODO motion planner requires to follow along a linear trajectory in workspace
def linear_trajectory_in_workspace(start_pose, end_pose, num_waypoints=100):
    # return np.linspace(start_pos, end_pos, num=num_waypoints)
    pass

def solve_ik_for_endpoints(physics, start_pose, end_pose, site_name='attachment_site'):
    
    start_xpos, start_quat = start_pose[:3], start_pose[3:]
    start_result = ik.qpos_from_site_pose(
        physics,
        site_name,
        start_xpos,
        target_quat=None)
    print(start_result)
    assert start_result.success, "IK failed for start pose."

    end_xpos, end_quat = end_pose[:3], end_pose[3:]
    end_result = ik.qpos_from_site_pose(
        physics,
        site_name,
        end_xpos,
        target_quat=None)
    
    assert end_result.success, "IK failed for end pose."
    
    return start_result.qpos, end_result.qpos

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
    
    b = np.transpose(np.column_stack([q0, v0, qf, vf]))

    c = np.linalg.solve(A, b)

    q_des = np.zeros((num_points, len(q0)))
    for i in range(len(q0)):
        q_des[:, i] = c[0, i] * t**3 + c[1, i] * t**2 + c[2, i] * t + c[3, i]
     
    return q_des

def generate_trajectory(physics, start_pose, end_pose, num_waypoints=100, duration=2.0, site_name='attachment_site'):    
    """
    start_pose: np.array([x, y, z, qx, qy, qz, qw])
    end_pose: np.array([x, y, z, qx, qy, qz, qw])
    """
    start_qpos, end_qpos = solve_ik_for_endpoints(
        physics, 
        start_pose, end_pose, 
        site_name)
    
    assert start_qpos is not None, "IK failed for start pose."
    assert end_qpos is not None, "IK failed for end pose."
    
    q0, qf = start_qpos, end_qpos
    v0 = np.zeros_like(q0) # set initial velocity to zero
    vf = np.zeros_like(qf) # set final velocity to zero
    
    joint_trajectory = cubic_polynomial_interpolation(
        q0, qf, v0, vf, 
        num_points=num_waypoints, 
        duration=duration)
    
    assert np.allclose(start_qpos, joint_trajectory[0]), "Start pose not equal to initial joint position."
    assert np.allclose(end_qpos, joint_trajectory[-1]), "End pose not equal to final joint position."

    return joint_trajectory

def test_ik(physics, target_pos=None, target_quat=None): 
    qpos = physics.data.qpos
    site_xpos = physics.data.site_xpos[-1,:]
    
    print("Initial qpos: ", qpos)
    print("Initial xpos: ", site_xpos)
    if target_pos is None:
        target_pos = site_xpos
    
    ik_result = ik.qpos_from_site_pose(physics, 'attachment_site', target_pos, target_quat=None)
    print("IK result: ", ik_result)    
    return ik_result, ik_result.success

def move_to_target(physics, target_pose, duration=2.0):
    start_xpos = physics.data.site_xpos[-1,:]
    start_xquat = physics.data.xquat[-1,:] 
    start_pose = np.concatenate([start_xpos, start_xquat])
    print("Start pose: ", start_pose)
    print("Target pose: ", target_pose)
    joint_trajectory = generate_trajectory(physics, start_pose, target_pose, duration=duration)
    
    for i, qpos in enumerate(joint_trajectory):
        physics.data.qpos[:] = qpos
        physics.step()
        physics.render()
        time.sleep(0.01)
        print(f"Time step {i}: {qpos}")
        # TODO Visualization of the robot

def main(): 
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ur10e_model = os.path.join(curr_dir, '../data/universal_robots_ur10e/scene.xml')
    physics = dm_mujoco.Physics.from_xml_path(ur10e_model)
    target_pose = get_random_pose() 
    move_to_target(physics, target_pose)

if __name__ == "__main__":
    main()