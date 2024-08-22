import json
import mujoco 
import numpy as np
import trimesh
from residual_observer import ResidualObserver
from scipy.spatial.transform import Rotation as R


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
        self.contact_thres = self.config['contact_thres']  # Threshold for contact detection
    
        # mesh path
        # TODO: change mesh path to pkg_path + mesh_relative_path_config
        mesh_path = "/home/brucekimrok/RoboticsProjects/evcharging_ws/data/universal_robots_ur10e/assets/rod.obj"
        self.mesh = trimesh.load(mesh_path)
        self.face_normals = self.mesh.face_normals.copy()  

        # Get the body ID for the rod (assuming it's attached to wrist_3_link)
        self.rod_body_id = self.model.body_name2id('wrist_3_link')

        # Get the mesh-to-rod transform from the XML
        self.mesh_to_rod_pos = np.array([0, 0.27, 0])
        self.mesh_to_rod_rot = R.from_quat([1, 1, 0, 0]).as_matrix()



        self.residual = ResidualObserver(self.physics, config_path)

        self.Sigma_meas = np.eye(6)
        self.Sigma_meas_inv = np.linalg.inv(self.Sigma_meas)
        self.Fc_vectors = self._compute_friction_cone_vectors()
        self.osqp_solver = self._setup_osqp_solver()

        self.has_contact = False
        self.particles_mesh_frame = np.zeros((self.nop,3))
        self.indices_mesh_frame = np.zeros(self.nop, dtype=int)

    def sample_particles_on_meshes(self):
        """
        Sample points on the end effector's mesh. 
        :return: A tuple containing the sampled points and the corresponding triangle indices.
        """
        points_new, mesh_indices_new = self.mesh.sample(self.nop, return_index=True)
        self.particles_mesh_frame, self.indices_mesh_frame = points_new, mesh_indices_new
    
    def from_mesh_to_world_frame(self):
        # First, transform from mesh frame to rod frame
        particles_rod_frame = np.dot(self.particles_mesh_frame, self.mesh_to_rod_rot.T) + self.mesh_to_rod_pos

        # Then, transform from rod frame to world frame
        particles_world_frame = np.zeros_like(particles_rod_frame)
        for i, particle in enumerate(particles_rod_frame):
            mujoco.mj_local2Global(self.model, self.data, 
                                   particles_world_frame[i], None, 
                                   particle, None, 
                                   self.rod_body_id, 0)

        return particles_world_frame
    
    def run_motion_model(self):
        points_new = np.zeros((self.nop, 3))
        indices_new = np.zeros(self.nop, dtype=int)

        normals = self.face_normals[self.indices_mesh_frame]
        print(normals)
        random_component = np.random.randn(self.nop, 3) * 0.001
        random_component += normals * 0.1
        points_new_floating = self.particles_mesh_frame + random_component
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
            d_j = np.linalg.norm(self.particles_mesh_frame[j] - locations[i])
            if d_j < d[j]:
                d[j] = d_j
                points_new[j] = locations[i]
                indices_new[j] = index_tri[i]

            all_indices[j] = j

        # If the projection of (an old point + noise) does not intersect the
        # mesh, use the old point
        for i, j in enumerate(all_indices):
            if i != j:
                points_new[i] = self.particles_mesh_frame[i]
                indices_new[i] = self.indices_mesh_frame[i]
    
        # update particles
        self.particles_mesh_frame = points_new
        self.indices_mesh_frame = indices_new

    def run_measurement_model(self, gamma_t, particles_world_frame, current_time):
        X_t_bar = []
        for i in range(self.nop):
            r_t = particles_world_frame[i]
            weight = self._compute_likelihood(gamma_t, r_t)
            X_t_bar.append((r_t, weight))        

    def set_particles_to_zero(self):
        self.particles_mesh_frame = np.zeros((self.nop,3))

    def run_contact_particle_filter(self, current_time):
        # if epsilon(t) = γ(t)T Σ^−1_meas γ(t) < contact_thres:
        #     Xt = ∅
        #     return Xt
        gamma_t, _ = self.residual.get_residual(current_time) 

        if gamma_t.T @ self.Sigma_meas_inv @ gamma_t < self.contact_thres: # TODO: adjust contact_threshold
            self.set_particles_to_zero()
            return
        
        if not self.has_contact:
            # if all elements are 0, then we initialize particles
            self.sample_particles_on_meshes()
            self.has_contact = True
        else:
            self.run_motion_model()

        particles_world_frame = self.from_mesh_to_world_frame() # TODO: Check if this is correct
        
        Xt_bar = self.run_measurement_model(gamma_t, particles_world_frame, current_time)

        Xt = self.importance_resample(Xt_bar)
        return Xt