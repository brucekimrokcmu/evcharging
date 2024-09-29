import json
import os
from controller import UR10eController
from trajectory_generator import generate_trajectory, get_random_pose
from visualization import Visualization
from residual_observer import ResidualObserver
from dm_control import mujoco as dm_mujoco
import numpy as np

def main():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(curr_dir, "../data/universal_robots_ur10e/scene.xml")
    config_path = os.path.join(curr_dir, "./config.json")

    physics = dm_mujoco.Physics.from_xml_path(model_path)

    controller = UR10eController(physics, config_path)
    
    # start_pose = get_random_pose()
    # TODO: set a fixed start pose (p, q) 
    start_pose = np.array([0.4, -0.2, 0.5, 0.707, 0.0, 0.707, 0.0])
    target_pose = np.array([0.6, 0.2, 0.5, 0.707, 0.0, 0.0, 0.707])
    controller.init_pose(start_pose)
    # target_pose = get_random_pose()
    duration = 5.0
    joint_trajectory = generate_trajectory(
        controller.physics, start_pose, target_pose, num_waypoints=500, duration=duration
    )

    visualizer = Visualization()

    visualizer.visualize_PID_control(controller, joint_trajectory, duration)


if __name__ == "__main__":
    main()