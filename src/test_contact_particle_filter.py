import json
import os
from controller import UR10eController
from trajectory_generator import generate_trajectory, get_random_pose
from visualization import Visualization
from residual_observer import ResidualObserver
from dm_control import mujoco as dm_mujoco
from contact_particle_filter import ContactParticleFilter

def main():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(curr_dir, "../data/universal_robots_ur10e/scene.xml")
    config_path = os.path.join(curr_dir, "./config.json")

    physics = dm_mujoco.Physics.from_xml_path(model_path)

    # Initialize the ContactParticleFilter
    contact_particle_filter = ContactParticleFilter(physics, config_path)

    controller = UR10eController(physics, config_path)
    start_pose = get_random_pose()
    controller.init_pose(start_pose)

    print("Initial Paritcles:\n", contact_particle_filter.particles)
    print("Initial Mesh Indices:\n", contact_particle_filter.indices_in_mesh)

    # Sample particles on the mesh
    contact_particle_filter.sample_particles_on_meshes()
    print("Sampled Points:\n", contact_particle_filter.particles)
    print("Sampled Mesh Indices:\n", contact_particle_filter.indices_in_mesh)

    # Update the end-effector pose
    contact_particle_filter.update_end_effector_pose()
    print("End-Effector Pose:\n", contact_particle_filter.end_effector_pose)
    print("End-Effector Orientation:\n", contact_particle_filter.end_effector_orientation)

    # Run the motion model
    points_new, indices_new = contact_particle_filter.run_motion_model()
    print("New Points After Motion Model:\n", points_new)
    print("New Indices After Motion Model:\n", indices_new)

    # Reset contact status
    contact_particle_filter.reset_contact()
    print("Has Contact Been Reset?:", contact_particle_filter.has_contact)

if __name__ == "__main__":
    main()