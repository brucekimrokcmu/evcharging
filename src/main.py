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

def generate_trajectory(physics, start_pos, start_quat, end_pos, end_quat, num_waypoints=100, duration=2.0):
    trajectory = []

    # Linear interpolation for workspace
    waypoint_positions = np.linspace(start_pos, end_pos, num_waypoints)
    if start_quat is not None and end_quat is not None:
        pass # TODO slertp interpolation for quaternions
    else:
        waypoint_quats = [None] * num_waypoints

    # Polynomial trajectory in joint space 
    t_start = 0.0
    t_end = duration
    dt = (t_end - t_start) / (num_waypoints - 1)

    t_i = 0
    t_f = duration
    A = np.array([
          [t_i**3, t_i**2, t_i, 1],
          [3*t_i**2, 2*t_i, 1, 0],
          [t_f**3, t_f**2, t_f, 1],
          [3*t_f**2, 2*t_f, 1, 0]
      ])

    for i in range(num_waypoints):
        t = t_start + i * dt

        result  = ik.qpos_from_site_pose(
            physics, 
            'attachment_site', 
            waypoint_positions[i], 
            waypoint_quats[i])

        if result.success:
            qpos = result.qpos
            if i == 0:
                qi = qpos
                dqi = np.zeros_like(qpos)
            elif i == num_waypoints - 1:
                qf = qpos
                dqf = np.zeros_like(qpos)

            b = np.hstack([qi, dqi, qf, dqf])
            c = np.linalg.solve(A, b)

            q_des = c[0] * t**3 + c[1] * t**2 + c[2] * t + c[3]
            trajectory.append(q_des)

        else:
            print("IK failed at waypoint ", i)
            break

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