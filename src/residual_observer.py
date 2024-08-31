import json
import mujoco 
import numpy as np

class ResidualObserver:
    def __init__(self, physics, config_path):
        self.physics = physics
        self.model = self.physics.model.ptr
        self.data = self.physics.data.ptr
        self.num_joints = self.model.nv

        # Load the configuration file
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.step_size = self.config['step_size']
        self.step_count = 0
        self._initialize_residual_observer()

    # def _initialize_residual_observer(self):
    #     self.gain_matrix = np.diag(self.config['diagonal_gain'] * np.ones(self.num_joints)) 
    #     self.residual = np.zeros(self.num_joints)
    #     self.integral = np.zeros(self.num_joints)
    #     self.prev_time = 0
    
    def _initialize_residual_observer(self):
        self.gain_matrix = np.diag(self.config['diagonal_gain'] * np.ones(self.num_joints)) 
        self.x = np.zeros(2 * self.num_joints)  # [integral, residual]
        self.start_time = None
        self.prev_time = None

    # def get_residual(self, current_time):
    #     """
    #     Estimate external torques using residual observer method.
    #     Requires only proprioceptive measures (q, q_dot) and current commanded input u.

    #     r(t) = KI (p - integral_0_t(Btau + C^T(q,v)v + r(s))ds).

    #     p_dot = tau + tau_ext - alpha(q, q_dot)
    #     r = K[p + integral(alpha - tau - r)dt]

    #     This method should be called at each timestep of the simulation.

    #     Args:
    #         current_time (float): The current simulation time.

    #     Returns:
    #         tuple: A tuple containing:
    #             - residual (np.array): The estimated external torque.
    #             - integral (np.array): The current integral value.

    #     """
    #     dt = current_time - self.prev_time
    #     if dt <= 0:
    #         return self.residual, self.integral

    #     tau = self.data.actuator_force
    #     alpha = self._compute_alpha()
    #     p = self._compute_generalized_momentum()

    #     self.integral += (alpha - tau - self.residual) * dt
    #     self.residual = self.gain_matrix @ (self.integral - p)
    #     self.prev_time = current_time

    #     return self.residual, self.integral

    def get_residual(self, current_time):
        """
        Estimate external torques using residual observer method without applying forces.
        """
        if self.start_time is None:
            self.start_time = current_time
            self.prev_time = current_time
            return self.x[self.num_joints:], self.x[:self.num_joints]

        dt = current_time - self.prev_time
        if dt <= 0:
            return self.x[self.num_joints:], self.x[:self.num_joints]

        # Compute required quantities
        tau = self.data.actuator_force
        alpha = self._compute_alpha()
        p = self._compute_generalized_momentum()

        # Compute observer dynamics
        dx = self._observer_dynamics(current_time, self.x, p, tau, alpha)

        # Update observer state using simple Euler integration
        # This is now done locally without modifying MuJoCo's state
        self.x += dx * dt

        self.prev_time = current_time

        return self.x[self.num_joints:], self.x[:self.num_joints]  # [residual, integral]


    def _observer_dynamics(self, t, x, p, tau, alpha):
        """Define the dynamics of the residual observer."""
        integral, residual = np.split(x, 2)
        d_integral = alpha - tau - residual
        d_residual = self.gain_matrix @ (integral + p) # TODO: check integral sign
        return np.concatenate([d_integral, d_residual])

    def reset(self):
        """Reset the observer state."""
        self.x = np.zeros(2 * self.num_joints)
        self.start_time = None
        self.prev_time = None

    def _compute_alpha(self):
        """
        alpha_i = g_i(q) - (1/2)q̇^T @ (∂M(q)/∂(q_i)) @ q̇, i = 1, 2, ... , n
        """
        alpha = np.zeros(self.num_joints)
        q = self.data.qpos
        g = self._compute_gravity_term()
        dq = self.data.qvel

        for i in range(self.num_joints):
            dM_dqi = self._finite_difference_partial_M_qi(i)
            alpha[i] = g[i] - 0.5 * dq.T @ dM_dqi @ dq
            
        return alpha

    def _compute_gravity_term(self):
        g = np.zeros(self.model.nv)
        gravity = self.model.opt.gravity

        jacp = np.zeros((3, self.model.nv))  # For translational Jacobian
        jacr = np.zeros((3, self.model.nv))  # For rotational Jacobian (unused)
        
        for i in range(1, self.model.nbody):  # Start from 1 to skip the world body
            mass = self.model.body_mass[i]
            mujoco.mj_jacBodyCom(self.model, self.data, jacp, jacr, i)
            force = mass * gravity
            g += jacp.T @ force

        return g

    # TODO: Look into MuJoCo's API on derivative funcs
    def _finite_difference_partial_M_qi(self, i, eps=1e-6):
        q = self.data.qpos.copy()
        M_pos = np.zeros((self.num_joints, self.num_joints))
        M_neg = np.zeros((self.num_joints, self.num_joints))

        q[i] += eps
        mujoco.mj_setState(self.model, self.data, np.array(q).reshape(-1, 1), mujoco.mjtState.mjSTATE_QPOS)
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_fullM(self.model, M_pos, self.data.qM)
        
        q[i] -= 2 * eps
        mujoco.mj_setState(self.model, self.data, np.array(q).reshape(-1, 1), mujoco.mjtState.mjSTATE_QPOS)
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_fullM(self.model, M_neg, self.data.qM)
        
        q[i] += eps
        mujoco.mj_setState(self.model, self.data, np.array(q).reshape(-1, 1), mujoco.mjtState.mjSTATE_QPOS)
        mujoco.mj_forward(self.model, self.data)

        dM_dqi = (M_pos - M_neg) / (2 * eps)
        
        return dM_dqi

    def _compute_generalized_momentum(self):        
        momentum = np.zeros(self.num_joints)
        mujoco.mj_mulM(self.model, self.data, momentum, self.data.qvel)

        return momentum
