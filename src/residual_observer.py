import json
import mujoco 
import os
import numpy as np

class ResidualObserver:
    def __init__(self, physics, config_path):
        self.physics = physics
        self.model = self.physics.model.ptr
        self.data = self.physics.data.ptr
        self.num_joints = self.model.nv
        self._load_config(config_path)
        self._initialize_observer()

    def _load_config(self, config):
        if isinstance(config, dict):
            self.config = config
        elif isinstance(config, (str, os.PathLike)):
            with open(config, 'r') as f:
                self.config = json.load(f)
        else:
            raise TypeError("config must be either a dictionary or a path to a JSON file")
        self.step_size = self.config['step_size']
        self.gain_matrix = np.diag(self.config['diagonal_gain'] * np.ones(self.num_joints))

    def _initialize_observer(self):
        self.x = np.zeros(2 * self.num_joints)  # [integral, residual]
        self.start_time = None
        self.prev_time = None

    def get_residual(self, current_time):
        if not self._is_initialized(current_time):
            return self.x[self.num_joints:], self.x[:self.num_joints]

        dt = current_time - self.prev_time
        if dt <= 0:
            return self.x[self.num_joints:], self.x[:self.num_joints]

        tau = self.data.actuator_force
        alpha = self._compute_alpha()
        p = self._compute_generalized_momentum()

        dx = self._observer_dynamics(self.x, p, tau, alpha)
        self.x += dx * dt
        self.prev_time = current_time

        return self.x[self.num_joints:], self.x[:self.num_joints]  # [residual, integral]

    def _is_initialized(self, current_time):
        if self.start_time is None:
            self.start_time = current_time
            self.prev_time = current_time
            return False
        return True

    def _observer_dynamics(self, x, p, tau, alpha):
        integral, residual = np.split(x, 2)
        d_integral = alpha - tau - residual
        d_residual = self.gain_matrix @ (integral + p) 
        return np.concatenate([d_integral, d_residual])

    def reset(self):
        self._initialize_observer()

    def _compute_alpha(self):
        alpha = np.zeros(self.num_joints)
        dq = self.data.qvel
        dM_dq = self._finite_difference_partial_M()

        for i in range(self.num_joints):
            alpha[i] = self.data.qfrc_bias[i] - 0.5 * dq.T @ dM_dq[i] @ dq

        return alpha

    def _finite_difference_partial_M(self, eps=1e-6):
        q = self.data.qpos.copy()
        dM_dq = np.zeros((self.num_joints, self.num_joints, self.num_joints))

        # Compute M at the current configuration
        M_current = np.zeros((self.num_joints, self.num_joints))
        mujoco.mj_fullM(self.model, M_current, self.data.qM)

        # Compute M for positive and negative perturbations
        for i in range(self.num_joints):
            q[i] += eps
            self._set_state(q)
            M_pos = np.zeros((self.num_joints, self.num_joints))
            mujoco.mj_fullM(self.model, M_pos, self.data.qM)
            
            q[i] -= 2 * eps
            self._set_state(q)
            M_neg = np.zeros((self.num_joints, self.num_joints))
            mujoco.mj_fullM(self.model, M_neg, self.data.qM)
            
            dM_dq[i] = (M_pos - M_neg) / (2 * eps)
            
            # Reset q[i]
            q[i] += eps

        # Reset to original state
        self._set_state(q)

        return dM_dq

    def _set_state(self, q):
        mujoco.mj_setState(self.model, self.data, np.array(q).reshape(-1, 1), mujoco.mjtState.mjSTATE_QPOS)
        mujoco.mj_forward(self.model, self.data)


    def _compute_generalized_momentum(self):        
        momentum = np.zeros(self.num_joints)
        mujoco.mj_mulM(self.model, self.data, momentum, self.data.qvel)
        return momentum