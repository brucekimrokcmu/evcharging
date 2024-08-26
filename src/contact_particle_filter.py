import os
import json
import mujoco 
import numpy as np
import trimesh
from residual_observer import ResidualObserver
from scipy.spatial.transform import Rotation as R
import osqp

class ContactParticleFilter:
    def __init__(self, physics, config_path):
        self._initialize_from_config(physics, config_path)
        self._setup_mesh()
        self._setup_transforms()
        self._setup_residual_observer()
        self._setup_friction_cone()
        self._initialize_particles()

    def _initialize_from_config(self, physics, config_path):
        self.physics = physics
        self.model = self.physics.model.ptr
        self.data = self.physics.data.ptr  
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.nop = self.config['number_of_particles']
        self.friction_coefficient = self.config["friction_coefficient"]
        self.contact_thres = self.config['contact_thres']
        self.n_friction_vectors = self.config.get('friction_cone_vectors', 5)

    def _setup_mesh(self):
        # TODO: change mesh path to pkg_path + mesh_relative_path_config
        mesh_path = "/home/brucekimrok/RoboticsProjects/evcharging_ws/data/universal_robots_ur10e/assets/rod.obj"
        # mesh_path = self.config.get('mesh_path', "/home/brucekimrok/RoboticsProjects/evcharging_ws/data/universal_robots_ur10e/assets/rod.obj")
        self.mesh = trimesh.load(mesh_path)
        self.face_normals = self.mesh.face_normals.copy()

    def _setup_transforms(self):
        self.rod_body_id = self.model.body_name2id('wrist_3_link')
        self.mesh_to_rod_pos = np.array([0, 0.27, 0])
        self.mesh_to_rod_rot = R.from_quat([1, 1, 0, 0]).as_matrix()

    def _setup_residual_observer(self):
        self.residual = ResidualObserver(self.physics, self.config)
        self.Sigma_meas = np.eye(6)
        self.Sigma_meas_inv = np.linalg.inv(self.Sigma_meas)

    def _setup_friction_cone(self):
        self.Fc_vectors = self._compute_friction_cone_vectors()
        self.osqp_solver = self._setup_osqp_solver()

    def _initialize_particles(self):
        self.has_contact = False
        self.particles_mesh_frame = np.zeros((self.nop, 3))
        self.indices_mesh_frame = np.zeros(self.nop, dtype=int)

    def _compute_friction_cone_vectors(self):
        mu = self.friction_coefficient
        n = self.n_friction_vectors
        vectors = [np.array([0, 0, 1])]
        angle_step = 2 * np.pi / (n - 1)
        for i in range(1, n):
            angle = i * angle_step
            x, y = mu * np.cos(angle), mu * np.sin(angle)
            vectors.append(np.array([x, y, 1]))
        return np.array(vectors).T / np.sqrt(1 + mu**2)

    def _setup_osqp_solver(self):
        n = self.n_friction_vectors
        solver = osqp.OSQP()
        solver.setup(P=np.eye(n), q=np.zeros(n), A=np.eye(n), 
                     l=np.zeros(n), u=np.inf*np.ones(n), verbose=False)
        return solver

    def sample_particles_on_meshes(self):
        self.particles_mesh_frame, self.indices_mesh_frame = self.mesh.sample(self.nop, return_index=True)

    def from_mesh_to_world_frame(self):
        particles_rod_frame = np.dot(self.particles_mesh_frame, self.mesh_to_rod_rot.T) + self.mesh_to_rod_pos
        particles_world_frame = np.zeros_like(particles_rod_frame)
        for i, particle in enumerate(particles_rod_frame):
            mujoco.mj_local2Global(self.model, self.data, particles_world_frame[i], None, 
                                   particle, None, self.rod_body_id, 0)
        return particles_world_frame

    def run_motion_model(self):
        normals = self.face_normals[self.indices_mesh_frame]
        random_component = np.random.randn(self.nop, 3) * 0.001 + normals * 0.1
        points_new_floating = self.particles_mesh_frame + random_component
        a = -np.sign(np.sum(random_component * normals, axis=1))

        locations, index_ray, index_tri = self.mesh.ray.intersects_location(
            ray_origins=points_new_floating,
            ray_directions=(normals.T * a).T
        )

        points_new, indices_new = self._process_intersections(locations, index_ray, index_tri)
        self.particles_mesh_frame, self.indices_mesh_frame = points_new, indices_new

    def _process_intersections(self, locations, index_ray, index_tri):
        points_new = np.zeros_like(self.particles_mesh_frame)
        indices_new = np.zeros_like(self.indices_mesh_frame)
        d = np.full(self.nop, np.inf)

        for i, j in enumerate(index_ray):
            d_j = np.linalg.norm(self.particles_mesh_frame[j] - locations[i])
            if d_j < d[j]:
                d[j] = d_j
                points_new[j] = locations[i]
                indices_new[j] = index_tri[i]

        mask = d == np.inf
        points_new[mask] = self.particles_mesh_frame[mask]
        indices_new[mask] = self.indices_mesh_frame[mask]

        return points_new, indices_new

    def run_measurement_model(self, gamma_t, particles_world_frame):
        X_t_bar = []
        for r_t in particles_world_frame:
            J_r = np.zeros((6, self.model.nv))
            mujoco.mj_jac(self.model, self.data, J_r[:3], J_r[3:], r_t, self.rod_body_id)
            qp_result = self._solve_qp(gamma_t, J_r)
            prob = np.exp(-0.5 * qp_result)
            X_t_bar.append((r_t, prob))
        return X_t_bar

    def _solve_qp(self, gamma, J_r):
        P = J_r.T @ self.Sigma_meas_inv @ J_r
        q = -gamma.T @ self.Sigma_meas_inv @ J_r
        G = -np.eye(self.n_friction_vectors)
        h = np.zeros(self.n_friction_vectors)
        
        result = self.osqp_solver.solve(P, q, G, h)
        
        if result.info.status != 'solved':
            return float('inf')
        
        alpha = result.x
        Fc = self.Fc_vectors @ alpha
        return (gamma - J_r.T @ Fc).T @ self.Sigma_meas_inv @ (gamma - J_r.T @ Fc)

    def run_contact_particle_filter(self, current_time):
        gamma_t, _ = self.residual.get_residual(current_time) 

        if gamma_t.T @ self.Sigma_meas_inv @ gamma_t < self.contact_thres:
            self.particles_mesh_frame = np.zeros((self.nop, 3))
            return None

        if not self.has_contact:
            self.sample_particles_on_meshes()
            self.has_contact = True
        else:
            self.run_motion_model()

        particles_world_frame = self.from_mesh_to_world_frame()
        Xt_bar = self.run_measurement_model(gamma_t, particles_world_frame)
        return self.importance_resample(Xt_bar)

    def importance_resample(self, Xt_bar):
        # Implement importance resampling here
        pass