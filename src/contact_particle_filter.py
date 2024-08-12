import json
import mujoco 
import numpy as np
import trimesh

class ContactParticleFilter:
    def __init__(self, physics, config_path):
        self.physics = physics
        self.model = self.physics.model.ptr
        self.data = self.physics.data.ptr  

        # Load the configuration file
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.nop = self.config['number_of_particles']      
        self.friction_coefficient = self.config["friction_coefficient"]

        self.particles = np.zeros((self.nop,3))
        self.f_W_particles = np.zeros_like(self.particles)
        self.particles_optimal_values = np.full(self.nop, -np.inf)

        self.indices_in_mesh = np.zeros(self.nop, dtype=int)
        self.has_contact = False

        # mesh path
        # TODO: change mesh path to pkg_path + mesh_relative_path_config
        mesh_path = "/home/brucekimrok/RoboticsProjects/evcharging_ws/data/universal_robots_ur10e/assets/rod.obj"
        self.mesh = trimesh.load(mesh_path)
        self.face_normals = self.mesh.face_normals.copy()  



    def run_motion_model():
        pass

    def run_measurement_model():
        pass

    def sample_particles_on_meshes(self):
        """
        Sample points on the end effector's mesh. 
        :return: A tuple containing the sampled points and the corresponding triangle indices.
        """
        points_new, mesh_indices_new = self.mesh.sample(self.nop, return_index=True)
        self.link_particle_indices_list = [list(range(self.nop))]
        
        return points_new, mesh_indices_new

    def reset_contact(self):
        self.has_contact = False

    def run_contact_particle_filter():
        pass