import os
import json
import mujoco
import numpy as np
import trimesh
from residual_observer import ResidualObserver
from scipy.spatial.transform import Rotation as R
import osqp
import xml.etree.ElementTree as ET

class ContactParticleFilter:
    def __init__(self, physics, config_path):
        self.physics = physics
        self.model = self.physics.model.ptr
        self.data = self.physics.data.ptr
        self._load_config(config_path)
        self._setup_mesh()
        self._parse_mesh_transform()
        self._setup_residual_observer()
        self._setup_friction_cone()
        self._initialize_particles()

    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.nop = self.config['number_of_particles']
        self.friction_coefficient = self.config["friction_coefficient"]
        self.contact_thres = self.config['contact_thres']
        self.n_friction_vectors = self.config.get('friction_cone_vectors', 5)
        self.rod_body_id = self.model.body_name2id('wrist_3_link')
        
        # Add base paths for mesh and XML
        self.base_path = os.path.dirname(os.path.dirname(config_path))
        self.mesh_path = os.path.join(self.base_path, 'data', 'universal_robots_ur10e', 'assets', 'rod.obj')
        self.xml_path = os.path.join(self.base_path, 'data', 'universal_robots_ur10e', 'ur10e.xml')

    def _setup_mesh(self):
        if not os.path.exists(self.mesh_path):
            raise FileNotFoundError(f"Mesh file not found: {self.mesh_path}")
        
        self.mesh = trimesh.load(self.mesh_path)
        self.face_normals = self.mesh.face_normals.copy()

    def _parse_mesh_transform(self):
        if not os.path.exists(self.xml_path):
            raise FileNotFoundError(f"XML file not found: {self.xml_path}")
        
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        
        mesh_elem = root.find(".//body[@name='wrist_3_link']/geom[@type='mesh']")
        
        if mesh_elem is not None:
            pos = mesh_elem.get('pos')
            quat = mesh_elem.get('quat')
            
            self.mesh_to_rod_pos = np.array([float(x) for x in pos.split()]) if pos else np.zeros(3)
            self.mesh_to_rod_rot = R.from_quat([float(x) for x in quat.split()]).as_matrix() if quat else np.eye(3)
        else:
            raise RuntimeError("Mesh element not found in XML. Aborting.")
        
    def _setup_residual_observer(self):
        self.residual = ResidualObserver(self.physics, self.config)
        nv = self.physics.model.nv
        self.Sigma_meas = np.eye(nv) * self.config["convariance_standard_deviation"]
        self.Sigma_meas_inv = np.linalg.inv(self.Sigma_meas)

    def _setup_friction_cone(self):
        self.Fc_vectors = self._compute_friction_cone_vectors()
        self.osqp_solver = self._setup_osqp_solver()

    def _initialize_particles(self):
        self.has_contact = False
        self.particles_mesh_frame = np.empty((self.nop, 3))
        self.indices_mesh_frame = np.empty(self.nop, dtype=int)

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

    def _from_mesh_to_world_frame(self):
        particles_world_frame = np.zeros_like(self.particles_mesh_frame)
        for i, particle in enumerate(self.particles_mesh_frame):
            particle_rod = mujoco.mj_local2Local(self.model, self.data, 
                                                particle, self.mesh_to_rod_pos, self.mesh_to_rod_rot, 
                                                np.zeros(3), np.eye(3), self.rod_body_id)
            
            mujoco.mj_local2Global(self.model, self.data, 
                                   particles_world_frame[i], None,
                                   particle_rod, None, 
                                   self.rod_body_id, 0)
        return particles_world_frame

    def _from_world_to_mesh_frame(self, particles_world_frame):
        particles_mesh_frame = np.zeros_like(particles_world_frame)
        for i, particle in enumerate(particles_world_frame):
            particle_rod = np.zeros(3)
            mujoco.mj_global2Local(self.model, self.data, particle_rod, None, particle, None, self.rod_body_id)
            particles_mesh_frame[i] = np.dot(particle_rod - self.mesh_to_rod_pos, self.mesh_to_rod_rot)
        return particles_mesh_frame

    def run_motion_model(self):
        normals = self.face_normals[self.indices_mesh_frame]
        random_component = np.random.randn(self.nop, 3) * 0.001 + normals * 0.1
        points_new_floating = self.particles_mesh_frame + random_component
        a = -np.sign(np.sum(random_component * normals, axis=1))

        locations, index_ray, index_tri = self.mesh.ray.intersects_location(
            ray_origins=points_new_floating,
            ray_directions=(normals.T * a).T
        )

        self.particles_mesh_frame, self.indices_mesh_frame = self._process_intersections(locations, index_ray, index_tri)

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
        nv = self.model.nv
        X_t_bar = []
        unnormalized_probs = []
        
        for r_t in particles_world_frame:
            J_r = np.zeros((3, nv))
            mujoco.mj_jac(self.model, self.data, J_r[:3], J_r[3:], r_t, self.rod_body_id)
            
            qp_result = self._solve_qp(gamma_t, J_r)
            unnormalized_prob = np.exp(-0.5 * qp_result)
            unnormalized_probs.append(unnormalized_prob)
            X_t_bar.append((r_t, unnormalized_prob))

        total_prob = sum(unnormalized_probs)
        if total_prob > 0:
            normalized_X_t_bar = [(r_t, prob / total_prob) for (r_t, prob) in X_t_bar]
        else:
            equal_prob = 1.0 / len(X_t_bar)
            normalized_X_t_bar = [(r_t, equal_prob) for (r_t, _) in X_t_bar]
        
        return normalized_X_t_bar

    def _solve_qp(self, gamma, J_r):
        P = J_r.T @ self.Sigma_meas_inv @ J_r
        q = -gamma.T @ self.Sigma_meas_inv @ J_r
        
        G = np.zeros((6, self.n_friction_vectors))
        G[:3] = -self.Fc_vectors
        G[3:] = -np.cross(self.Fc_vectors.T, np.array([1, 0, 0])).T
        h = np.zeros(6)
        
        result = self.osqp_solver.solve(P, q, G, h)
        
        if result.info.status != 'solved':
            return float('inf')
        
        alpha = result.x
        Fc = np.zeros(6)
        Fc[:3] = self.Fc_vectors @ alpha
        Fc[3:] = np.cross(Fc[:3], np.array([1, 0, 0]))
        return (gamma - J_r.T @ Fc).T @ self.Sigma_meas_inv @ (gamma - J_r.T @ Fc)
  
    def run_contact_particle_filter(self, current_time):
        gamma_t, _ = self.residual.get_residual(current_time) 
        e_t = gamma_t.T @ self.Sigma_meas_inv @ gamma_t

        if e_t < self.contact_thres:
            self.particles_mesh_frame = np.zeros((self.nop, 3))
            self.has_contact = False
            return self.particles_mesh_frame

        if not self.has_contact:
            self.sample_particles_on_meshes()
            self.has_contact = True
        else:
            self.run_motion_model()

        particles_world_frame = self._from_mesh_to_world_frame()
        normalized_Xt_bar = self.run_measurement_model(gamma_t, particles_world_frame)
        return self.importance_resample(normalized_Xt_bar)

    def importance_resample(self, normalized_Xt_bar):
        particles, weights = zip(*normalized_Xt_bar)
        particles = np.array(particles)
        weights = np.array(weights)

        indices = np.random.choice(len(particles), size=self.nop, p=weights)
        resampled_particles = particles[indices]

        self.particles_mesh_frame = self._from_world_to_mesh_frame(resampled_particles)
        return self.particles_mesh_frame

    def get_contact_points(self):
        particles_rod_frame = np.dot(self.particles_mesh_frame, self.mesh_to_rod_rot.T) + self.mesh_to_rod_pos
        particles_world_frame = self._from_mesh_to_world_frame()
        return particles_rod_frame, particles_world_frame