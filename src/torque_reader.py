import mujoco 
from dm_control import mujoco as dm_mujoco
import numpy as np

class TorqueReader:
    def __init__(self, model_path, config):
        self.physics = dm_mujoco.Physics.from_xml_path(model_path)
        self.model = self.physics.model.ptr
        self.data = self.physics.data.ptr
        self.num_joints = self.model.nv

        self.config = config
        self.step_size = self.config['step_size']
        self.step_count = 0
        self.initialize_residual_observer()

    def initilize_residual_observer(self):
        self.gain_matirx = np.diag(self.config['diagonal_gain'] * self.num_joints)
        self.residual = np.zeros(self.num_joints)
        self.integral = np.zeros(self.num_joints)
        self.prev_time = 0

    def step(self):
        mujoco.mj_step(self.model, self.data)
    
    def estimate_external_torques(self):
        """
        Estimate external torques using residual observer method.
        Requires only proprioceptive measures (q, q_dot) and current commanded input u.
        """
        dt = self._calcualte_time_step()
        u = self.data.ctrl[:self.num_joints]

        alpha = self._compute_dynamic_effects()
        p = self._compute_generalized_momentum()

        self.integral += (alpha - u - self.residual) * dt
        self.residual = self.gain_matirx @ (self.integral + p)

        estimated_torques = self.residual / dt

        return estimated_torques
    
    def _calculate_time_step(self):
        current_time = self.step_count * self.step_size
        dt = current_time - self.prev_time
        self.prev_time = current_time
        return dt

    def _compute_dynamic_effects(self):
        dynamic_effects = self.data.qfrc_bais[:self.num_joints]
        return dynamic_effects

    def _compute_generalized_momentum(self):
        mujoco.mj_mulM(self.model, self.data, self.data.qM, self.data.qvel)
        return self.data.qM[:self.num_joints]

    # TODO: compare with below 
    # data.joint("my_joint").qfrc_constraint + data.joint("my_joint").qfrc_smooth