import numpy as np
from dm_control.utils import inverse_kinematics as ik

def get_random_pose(qpos_max=1.0, qpos_min=0.1):
    random_pos = np.random.uniform(qpos_min, qpos_max, 3)
    random_quaternion = np.random.normal(0, 1, 4)
    random_quaternion /= np.linalg.norm(random_quaternion)
    random_pose = np.concatenate([random_pos, random_quaternion])
    return random_pose

def solve_ik_for_endpoints(physics, start_pose, end_pose, site_name="attachment_site"):
    start_xpos, start_quat = start_pose[:3], start_pose[3:]
    start_result = ik.qpos_from_site_pose(
        physics, site_name, start_xpos, target_quat=None
    )
    assert start_result.success, "IK failed for start pose."

    end_xpos, end_quat = end_pose[:3], end_pose[3:]
    end_result = ik.qpos_from_site_pose(physics, site_name, end_xpos, target_quat=None)
    assert end_result.success, "IK failed for end pose."

    return start_result.qpos, end_result.qpos

def cubic_polynomial_interpolation(q0, qf, v0=None, vf=None, num_points=500, duration=5.0):
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
        q_des[:, i] = c[0, i]*t**3 + c[1, i]*t**2 + c[2, i]*t + c[3, i]

    return q_des

def generate_trajectory(physics, start_pose, end_pose, num_waypoints=500, duration=5.0, site_name="attachment_site"):
    start_qpos, end_qpos = solve_ik_for_endpoints(physics, start_pose, end_pose, site_name)

    assert start_qpos is not None, "IK failed for start pose."
    assert end_qpos is not None, "IK failed for end pose."

    q0, qf = start_qpos, end_qpos
    v0 = np.zeros_like(q0)
    vf = np.zeros_like(qf)

    joint_trajectory = cubic_polynomial_interpolation(q0, qf, v0, vf, num_points=num_waypoints, duration=duration)

    assert np.allclose(start_qpos, joint_trajectory[0]), "Start qpos not equal to initial joint position."
    assert np.allclose(end_qpos, joint_trajectory[-1]), "End qpos not equal to final joint position."

    return joint_trajectory