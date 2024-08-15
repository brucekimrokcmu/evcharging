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
        self.indices_in_mesh = np.zeros(self.nop, dtype=int)
        self.has_contact = False

    
        # mesh path
        # TODO: change mesh path to pkg_path + mesh_relative_path_config
        mesh_path = "/home/brucekimrok/RoboticsProjects/evcharging_ws/data/universal_robots_ur10e/assets/rod.obj"
        self.mesh = trimesh.load(mesh_path)
        self.face_normals = self.mesh.face_normals.copy()  

        self.Σmotion = np.diag([0.001, 0.001, 0.001])  # Adjust these values as needed

    def run_motion_model(self):
        points_new = np.zeros((self.nop, 3))
        indices_new = np.zeros(self.nop, dtype=int)

        normals = self.face_normals[self.indices_in_mesh]
        random_component = np.random.randn(self.nop, 3) * 0.001
        random_component += normals * 0.1
        points_new_floating = self.particles + random_component
        a = -np.sign(np.sum(random_component * normals, axis=1))  # correct sign

        # ray casting
        locations, index_ray, index_tri = self.mesh.ray.intersects_location(
            ray_origins=points_new_floating,
            ray_directions=(normals.T * a).T
        )

        # create points_new projected and indices_new
        d = np.full(self.nop, np.inf)
        all_indices = np.full(self.nop, self.nop * 2)

        for i, j in enumerate(index_ray):
            d_j = np.linalg.norm(self.particles[j] - locations[i])
            if d_j < d[j]:
                d[j] = d_j
                points_new[j] = locations[i]
                indices_new[j] = index_tri[i]

            all_indices[j] = j

        # If the projection of (an old point + noise) does not intersect the
        # mesh, use the old point
        for i, j in enumerate(all_indices):
            if i != j:
                points_new[i] = self.particles[i]
                indices_new[i] = self.indices_in_mesh[i]

        return points_new, indices_new

    def run_measurement_model():
        pass

    def sample_particles_on_meshes(self):
        """
        Sample points on the end effector's mesh. 
        :return: A tuple containing the sampled points and the corresponding triangle indices.
        """
        points_new, mesh_indices_new = self.mesh.sample(self.nop, return_index=True)
        
        return points_new, mesh_indices_new

    def update_end_effector_pose(self):
        self.end_effector_pose = self.data.site_xpos.copy()
        self.end_effector_orientation = self.data.site_xmat.reshape(3, 3).copy()

    def reset_contact(self):
        self.has_contact = False

    def run_contact_particle_filter():
        pass