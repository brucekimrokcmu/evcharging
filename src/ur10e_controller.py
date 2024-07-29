import mujoco
from dm_control import mujoco as dm_mujoco
from dm_control.utils import inverse_kinematics as ik
import numpy as np

class UR10eController:
    def __init__(self, model_path):
        self.physics = dm_mujoco.Physics.from_xml_path(model_path)
        self.model = self.physics.model.ptr
        self.data = self.physics.data.ptr

    def init_pose(self, pose, site_name="rod_tip"):
        xpos, quat = pose[:3], pose[3:]
        ik_result = ik.qpos_from_site_pose(
            self.physics, site_name, xpos, target_quat=quat, inplace=True
        )
        assert ik_result.success, "IK failed for initial pose."

    def set_joint_positions(self, qpos):
        self.data.qpos[:] = qpos

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def get_end_effector_position(self):
        return self.data.site_xpos[-1, :]
    
    def get_time(self):
        return self.data.time