import json
import os
from ur10e_controller import UR10eController
from trajectory_generator import generate_trajectory, get_random_pose
from visualization import Visualization
from residual_observer import ResidualObserver

def main():
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ur10e_model = os.path.join(curr_dir, "../data/universal_robots_ur10e/scene.xml")

    controller = UR10eController(ur10e_model)
    
    pose_start = get_random_pose()
    controller.init_pose(pose_start)

    target_pose = get_random_pose()

    duration = 5.0
    joint_trajectory = generate_trajectory(
        controller.physics, pose_start, target_pose, num_waypoints=500, duration=duration
    )

    visualizer = Visualization()

    visualizer.visualize_trajectory(controller, joint_trajectory, duration)



if __name__ == "__main__":
    main()