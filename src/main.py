import os
import time
import numpy as np
import mujoco
import mujoco.viewer as viewer
from IPython.display import clear_output

#TODO Renderer, Camera

SIM_DURATION = 5
MODEL_PATH = '../data/universal_robots_ur10e/scene.xml'

"""
MuJoCO has mj_getState, and mj_setState. 
"""

def get_joint_values(model, data):
    """
    Get the joint values(qpos) from the MuJoCo data object
    
    Args:
        model (mjModel): MuJoCo model object
        data (mjData): MuJoCo data object
    
    Returns:
        np.ndarray: Numpy array containing the joint values
        
    """
    qpos_array = np.zeros(model.nq, dtype=np.float64)
    mujoco.mj_getState(model, data, qpos_array, mujoco.mjtState.mjSTATE_QPOS)
    
    return qpos_array


def get_pose(model, data):
    """
    Get the body pose(xpos) from the MuJoCo data object
    
    Args:
        model (mjModel): MuJoCo model object
        data (mjData): MuJoCo data object
    
    Returns:
        mjData.xpos: MuJoCo data object containing the body pose
        
    """

    mujoco.mj_forward(model, data)


    return np.copy(np.reshape(data.xpos, (model.nbody, 3)))

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

    # Test get_joint_values
    joint_values = get_joint_values(model, data).copy()
    print("Initial joint values: ", joint_values)

    body_pose = get_pose(model, data)
    print("Initial body pose: ", body_pose)

    initial_joint_values = joint_values.copy()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance *= 2.5

        start = time.time()
        while viewer.is_running() and time.time() - start < SIM_DURATION:
            step_start = time.time()
        
            mujoco.mj_step(model, data)

            elapsed_time = time.time() - start

            # joint_values = get_joint_values(model, data)
            body_pose = get_pose(model, data)
            # print(f"\nJoint values at time {elapsed_time:.2f}: {joint_values}")
            print(f"Body pose at time {elapsed_time:.2f}: ")
            print(body_pose)


            # TODO: use MuJoCo's mju_add or simple mycontroller to add the joint values 
            data.qpos = initial_joint_values + (-2 * elapsed_time / SIM_DURATION)

            # with viewer.lock():
                # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time %2) # Toggle contact points
            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)  
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()