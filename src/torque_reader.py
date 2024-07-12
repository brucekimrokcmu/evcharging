import mujoco 
from dm_control import mujoco as dm_mujoco
import numpy as np

class TorqueReader:
    def __init__(self, model_path):
        self.physics = dm_mujoco.Physics.from_xml_path(model_path)
        self.model = self.physics.model.ptr
        self.data = self.physics.data.ptr
        self.nv = self.model.nv

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

    def step(self):
        mujoco.mj_step(self.model, self.data)
    
    def get_joint_torques(self):
        mujoco.mj_inverse(self.model, self.data)

        # Get different torque components
        qfrc_inverse = self.data.qfrc_inverse.copy()  # Main output of inverse dynamics
        qfrc_applied = self.data.qfrc_applied.copy()  # Applied forces
        qfrc_actuator = self.data.actuator_force.copy()  # Actuator forces

        # Compute Jacobian'*xfrc_applied
        jacp = np.zeros((3 * self.model.nbody, self.model.nv))
        jacr = np.zeros((3 * self.model.nbody, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, -1)
        xfrc_applied = self.data.xfrc_applied.reshape(-1)
        jac_xfrc = np.dot(jacp.T, xfrc_applied[:self.model.nbody*3]) + np.dot(jacr.T, xfrc_applied[self.model.nbody*3:])

        # Compute external forces (any unexplained force)
        qfrc_external = qfrc_inverse - qfrc_applied - jac_xfrc - qfrc_actuator

        return {
            'inverse': qfrc_inverse,
            'applied': qfrc_applied,
            'actuator': qfrc_actuator,
            'jac_xfrc': jac_xfrc,
            'external': qfrc_external
        }