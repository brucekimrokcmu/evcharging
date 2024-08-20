import json
import mujoco 
import numpy as np
import trimesh
from residual_observer import ResidualObserver
import osqp
import utils
from utils import *

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
        self.epsilon_bar = self.config['epsilon_bar']  # Threshold for contact detection
    
        # mesh path
        # TODO: change mesh path to pkg_path + mesh_relative_path_config
        mesh_path = "/home/brucekimrok/RoboticsProjects/evcharging_ws/data/universal_robots_ur10e/assets/rod.obj"
        self.mesh = trimesh.load(mesh_path)
        self.face_normals = self.mesh.face_normals.copy()  
    
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

    def _from_mesh_to_world_frame(self, body_name='rod_contact'):         
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            body_pos = self.data.xpos[body_id]
            body_quat = self.data.xquat[body_id]

            body_rot_matrix = mujoco.mju_quat2mat(body_quat)
            particles_world_frame = np.dot(self.particles_mesh_frame, body_rot_matrix.T) + body_pos
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
        # if epsilon(t) = γ(t)T Σ^−1_meas γ(t) < epsilon_bar:
        #     Xt = ∅
        #     return Xt
        gamma_t, _ = self.residual.get_residual(current_time)

        if gamma_t @ self.Sigma_meas_inv @ gamma_t.T < self.epsilon_bar: # No contact # TODO: Check if this is correct: need a test script
            self.set_particles_to_zero()
            return
        
        if not self.has_contact:
            # if all elements are 0, then we initialize particles
            self.sample_particles_on_meshes()
            self.has_contact = True
        else:
            self.run_motion_model()

        particles_world_frame = self._from_mesh_to_world_frame() # TODO: Check if this is correct
        
        Xt_bar = self.run_measurement_model(gamma_t, particles_world_frame, current_time)

        Xt = self.importance_resample(Xt_bar)
        return Xt