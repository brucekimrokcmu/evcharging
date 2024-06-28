import os
import time
import numpy as np
from dm_control import mujoco as dm_mujoco
from dm_control.mujoco.wrapper.mjbindings import mjlib, enums
from dm_control.utils import inverse_kinematics as ik
import glfw
import mujoco.viewer 

print_camera_config = 0

#TODO: explore diverse IKs - trac ik, fast ik

def get_random_pose(qpos_max=1.0, qpos_min=0.1):
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
    assert start_result.success, "IK failed for start pose."

    end_xpos, end_quat = end_pose[:3], end_pose[3:]
    end_result = ik.qpos_from_site_pose(
        physics,
        site_name,
        end_xpos,
        target_quat=None)
    
    assert end_result.success, "IK failed for end pose."
    
    return start_result.qpos, end_result.qpos

# TODO: adjust num_points and duration in accordance to the distance between q0 and qf
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
        q_des[:, i] = c[0, i] * t**3 + c[1, i] * t**2 + c[2, i] * t + c[3, i]
     
    return q_des

def generate_trajectory(physics, start_pose, end_pose, num_waypoints=500, duration=5.0, site_name='attachment_site'):    
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
    
    assert np.allclose(start_qpos, joint_trajectory[0]), "Start qpos not equal to initial joint position."
    assert np.allclose(end_qpos, joint_trajectory[-1]), "End qpos not equal to final joint position."

    return joint_trajectory


def init_pose(physics, pose, site_name='attachment_site'):
    xpos, quat = pose[:3], pose[3:]
    ik_result = ik.qpos_from_site_pose(
            physics,
            site_name,
            xpos,
            target_quat=None,
            inplace=True
            )
    assert ik_result.success, "IK failed for initial pose."

def main(): 
    # Load UR10e model
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ur10e_model = os.path.join(curr_dir, '../data/universal_robots_ur10e/scene.xml')
    
    # Initialize physics and MuJoCo data structures
    physics = dm_mujoco.Physics.from_xml_path(ur10e_model)    
    model = physics.model.ptr
    data = physics.data.ptr   
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()

    # Initialize GLFW
    glfw.init()
    window = glfw.create_window(800, 800, "UR10e", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # Initialize visualization data structures
    mujoco.mjv_defaultCamera(cam)
    mujoco.mjv_defaultOption(opt)
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

    cam.azimuth = 89.83044433593757 ; cam.elevation = -89.0 ; cam.distance =  5.04038754800176
    cam.lookat =np.array([ 0.0 , 0.0 , 0.0 ])

    SIM_DUR = 10000
    sim_time = 0
    dt = 0.001

    # Launch the viewer
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        # Set camera configuration
        viewer.cam.azimuth = 89.83044433593757
        viewer.cam.elevation = -45.0
        viewer.cam.distance = 5.04038754800176
        viewer.cam.lookat = np.array([0.0, 0.0, 0.5])

        # Set initial pose
        pose_start = get_random_pose() #TODO: make this as config
        init_pose(physics, pose_start)

        # Set target pose 
        target_pose = get_random_pose()

        # Generate trajectory
        joint_trajectory = generate_trajectory(physics, pose_start, target_pose, num_waypoints=50000, duration=5.0)
        num_steps = joint_trajectory.shape[0]

        start_time = time.time()
        step = 0
        while viewer.is_running() and step < num_steps and time.time() - start_time < 5:
            time_prev = sim_time
            while (sim_time - time_prev < 1.0/60.0):

                ################################################################
                # TODO: Update the joint positions
                if step < num_steps:
                    data.qpos[:] = joint_trajectory[step]
                    step += 1

                ################################################################    
                mujoco.mj_step(model, data)
                sim_time += dt

            print(f"Step: {step}, End effector position: {data.site_xpos[-1,:]}")
            # Update the viewer
            viewer.sync()

    print("Visualization complete.")

if __name__ == "__main__":
    main()
