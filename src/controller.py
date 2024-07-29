import mujoco
from dm_control import mujoco as dm_mujoco
from dm_control.utils import inverse_kinematics as ik
import numpy as np
import json

class UR10eController:
    def __init__(self, model_path, config_path):
        self.physics = dm_mujoco.Physics.from_xml_path(model_path)
        self.model = self.physics.model.ptr
        self.data = self.physics.data.ptr

        with open(config_path, 'r') as f:
            config = json.load(f)
            self.kp = config['kp']
            self.ki = config['ki']
            self.kd = config['kd']

        self.previous_error = np.zeros(self.model.nv)
        self.integral = np.zeros(self.model.nv)
        self.dt = self.model.opt.timestep

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

    def apply_pid_control(self, joint_trajectory, dt=0.01):
        num_points = joint_trajectory.shape[0]
        num_joints = joint_trajectory.shape[1]
        applied_torques = np.zeros((num_points, num_joints))

        for i in range(num_points):
            desired_qpos = joint_trajectory[i]
            actual_qpos = self.data.qpos[:num_joints]
            torques = self.pid_controller.compute(desired_qpos, actual_qpos)
            self.data.ctrl[:num_joints] = torques
            self.step()
            applied_torques[i] = torques

        return applied_torques

    def compute_pid_torques(self, desired, actual):
        error = desired - actual
        self.integral += error * self.dt
        derivative = (error - self.previous_error) / self.dt
        self.previous_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative
