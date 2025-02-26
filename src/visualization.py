import time
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
from contact_particle_filter import ContactParticleFilter

class Visualization:
    def __init__(self):
        pass

    def visualize_trajectory(self, controller, joint_trajectory, duration=5.0):
        num_steps = joint_trajectory.shape[0]
        step_duration = duration / num_steps
        
        with mujoco.viewer.launch_passive(
            controller.model, controller.data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            # Set camera configuration
            viewer.cam.azimuth = 89.83044433593757
            viewer.cam.elevation = -45.0
            viewer.cam.distance = 5.04038754800176
            viewer.cam.lookat = [0.0, 0.0, 0.5]

            start_time = time.time()
            step = 0
            last_update_time = start_time

            while viewer.is_running() and step < num_steps:
                current_time = time.time()
                elapsed_time = current_time - start_time

                if elapsed_time >= step * step_duration:
                    if step < num_steps:
                        controller.set_joint_positions(joint_trajectory[step])
                        print(f"Step: {step}, End effector position: {controller.get_end_effector_position()}")
                        step += 1

                        controller.step()

                if current_time - last_update_time >= 1/60:  # Cap at 60 FPS
                    viewer.sync()
                    last_update_time = current_time

                # Sleep to maintain real-time simulation
                time_to_sleep = (start_time + step * step_duration) - time.time()
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)

        print("Visualization complete.")

    def visualize_PID_control(self, controller, joint_trajectory, duration=5.0):
        num_steps = joint_trajectory.shape[0]
        step_duration = duration / num_steps
        
        time_points = []
        desired_positions = []
        actual_positions = []
        control_signals = []
        position_errors = []
        target_end_effector_positions = []
        actual_end_effector_positions = []


        with mujoco.viewer.launch_passive(
            controller.model, controller.data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            # Set camera configuration
            viewer.cam.azimuth = 89.83044433593757
            viewer.cam.elevation = -45.0
            viewer.cam.distance = 5.04038754800176
            viewer.cam.lookat = [0.0, 0.0, 0.5]

            start_time = time.time()
            step = 0
            last_update_time = start_time

            while viewer.is_running() and step < num_steps:
                current_time = time.time()
                elapsed_time = current_time - start_time

                if elapsed_time >= step * step_duration:
                    if step < num_steps:
                        desired_qpos = joint_trajectory[step]
                        actual_qpos = controller.data.qpos
                        torques = controller.compute_pid_torques(desired_qpos, actual_qpos)
                        controller.data.ctrl = torques
                        controller.forward()
                        controller.step()
                        target_end_effector_pos = controller.data.site_xpos.copy()
                        actual_end_effector_pos = controller.get_end_effector_position()

                        # Record data after stepping the simulation
                        time_points.append(elapsed_time)
                        desired_positions.append(desired_qpos)
                        actual_positions.append(controller.data.qpos)
                        control_signals.append(torques)
                        position_errors.append(desired_qpos - controller.data.qpos[:len(desired_qpos)])
                        target_end_effector_positions.append(target_end_effector_pos)
                        actual_end_effector_positions.append(actual_end_effector_pos)

                        print(f"Step: {step}")
                        print(f"Desired position: {desired_qpos}")
                        print(f"Actual position: {controller.data.qpos[:len(desired_qpos)]}")
                        print(f"Control signal: {torques}")
                        print(f"Position error: {desired_qpos - controller.data.qpos[:len(desired_qpos)]}")
                        print(f"End effector position: {controller.get_end_effector_position()}")
                        print("---")

                        step += 1

                if current_time - last_update_time >= 1/60:  # Cap at 60 FPS
                    viewer.sync()
                    last_update_time = current_time

                time_to_sleep = (start_time + step * step_duration) - time.time()
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)

        print("Visualization complete.")

        time_points = np.array(time_points)
        desired_positions = np.array(desired_positions)
        actual_positions = np.array(actual_positions)
        control_signals = np.array(control_signals)
        position_errors = np.array(position_errors)        
        target_end_effector_positions = np.array(target_end_effector_positions)
        actual_end_effector_positions = np.array(actual_end_effector_positions)

        # Plot the results
        num_joints = desired_positions.shape[1]
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        fig, ax = plt.subplots(figsize=(12, 8))

        # Control signals
        for i in range(num_joints):
            axes[0].plot(time_points, control_signals[:, i], label=f'Joint {i+1}')
        axes[0].set_ylabel('Control Signals (Torques)')
        axes[0].legend()
        axes[0].grid(True)
        
        # Position errors
        for i in range(num_joints):
            axes[1].plot(time_points, position_errors[:, i], label=f'Joint {i+1}')
        axes[1].set_ylabel('Position Errors')
        axes[1].set_xlabel('Time (s)')
        axes[1].legend()
        axes[1].grid(True)
    
        # End-effector position errors
        end_effector_errors = target_end_effector_positions - actual_end_effector_positions
        axes[2].plot(time_points, end_effector_errors[:, 0], label='X Error')
        axes[2].plot(time_points, end_effector_errors[:, 1], label='Y Error')
        axes[2].plot(time_points, end_effector_errors[:, 2], label='Z Error')
        axes[2].set_ylabel('End Effector Position Errors')
        axes[2].set_xlabel('Time (s)')
        axes[2].legend()
        axes[2].grid(True)

        end_effector_errors = target_end_effector_positions - actual_end_effector_positions
        ax.plot(time_points, end_effector_errors[:, 0], label='X Error')
        ax.plot(time_points, end_effector_errors[:, 1], label='Y Error')
        ax.plot(time_points, end_effector_errors[:, 2], label='Z Error')
        ax.set_ylabel('End Effector Position Errors')
        ax.set_xlabel('Time (s)')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.show()

    def visualize_residual_observer(self, controller, joint_trajectory, observer, duration=5.0):
        num_steps = joint_trajectory.shape[0]
        step_duration = duration / num_steps
        
        time_steps = []
        ctrl_data = []
        qvel_data = []
        external_data = []
        residual_data = []
        integral_data = []

        with mujoco.viewer.launch_passive(
            controller.model, controller.data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            # Set camera configuration
            viewer.cam.azimuth = 89.83044433593757
            viewer.cam.elevation = -45.0
            viewer.cam.distance = 5.04038754800176
            viewer.cam.lookat = [0.0, 0.0, 0.5]

            start_time = time.time()
            sim_time = 0
            step = 0
            last_update_time = start_time

            while viewer.is_running() and step < num_steps:
                current_time = time.time()
                elapsed_time = current_time - start_time

                if elapsed_time >= step * step_duration:
                    if step < num_steps:
                        desired_qpos = joint_trajectory[step]
                        actual_qpos = controller.data.qpos
                        torques = controller.compute_pid_torques(desired_qpos, actual_qpos)
                        controller.data.ctrl = torques
                        
                        mujoco.mj_step(controller.model, controller.data)
                        sim_time += controller.model.opt.timestep
                        
                        residual, integral = observer.get_residual(sim_time)
                        
                        ctrl = controller.data.ctrl.copy()
                        qvel = controller.data.qvel.copy()
                        external_wo_actuator = controller.data.qfrc_constraint
                        
                        time_steps.append(sim_time)
                        ctrl_data.append(ctrl)
                        qvel_data.append(qvel)
                        residual_data.append(residual)
                        integral_data.append(integral)
                        external_data.append(external_wo_actuator)

                        step += 1

                        # Print progress every 100 steps
                        if step % 100 == 0:
                            print(f"Step: {step}/{num_steps}, Sim Time: {sim_time:.3f}")

                if current_time - last_update_time >= 1/60:  # Cap at 60 FPS
                    viewer.sync()
                    last_update_time = current_time

                # Sleep to maintain real-time simulation
                time_to_sleep = (start_time + step * step_duration) - time.time()
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)

        print("Visualization complete.")
        
        # Plot the residual and external constraint force for each joint
        num_joints = controller.model.nv
        fig, axs = plt.subplots(num_joints, 2, figsize=(18, 6 * num_joints))

        for i in range(num_joints):
            # Plot residual
            axs[i, 0].plot(time_steps, [residual[i] for residual in residual_data], label='Residual')
            axs[i, 0].set_xlabel('Time (s)')
            axs[i, 0].set_ylabel('Residual')
            axs[i, 0].set_title(f'Residual for Joint {i}')
            axs[i, 0].legend()
            axs[i, 0].grid()

            # Plot external constraint force
            axs[i, 1].plot(time_steps, [ext[i] for ext in external_data], label='External Force w/o actuator')
            axs[i, 1].set_xlabel('Time (s)')
            axs[i, 1].set_ylabel('External Force w/o actuator')
            axs[i, 1].set_title(f'External Force w/o actuator for Joint {i}')
            axs[i, 1].legend()
            axs[i, 1].grid()

        plt.tight_layout()
        plt.show()

    def visualize_cpf(self, controller, joint_trajectory, cpf, duration=5.0):
        if not isinstance(cpf, ContactParticleFilter):
            raise TypeError("cpf must be an instance of ContactParticleFilter")
        num_steps = joint_trajectory.shape[0]
        step_duration = duration / num_steps
        time_steps = []
        contact_detected = []
        num_particles = []
        contact_positions = []
        residual_magnitudes = []

        with mujoco.viewer.launch_passive(
            controller.model, controller.data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            # Set camera configuration
            viewer.cam.azimuth = 89.83044433593757
            viewer.cam.elevation = -45.0
            viewer.cam.distance = 5.04038754800176
            viewer.cam.lookat = [0.0, 0.0, 0.5]

            start_time = time.time()
            sim_time = 0
            step = 0
            last_update_time = start_time

            while viewer.is_running() and step < num_steps:
                current_time = time.time()
                elapsed_time = current_time - start_time

                if elapsed_time >= step * step_duration:
                    if step < num_steps:
                        desired_qpos = joint_trajectory[step]
                        controller.set_joint_positions(desired_qpos)
                        mujoco.mj_step(controller.model, controller.data)

                        sim_time += controller.model.opt.timestep
                        result = cpf.run_contact_particle_filter(sim_time)
                        gamma_t, _ = cpf.residual.get_residual(sim_time)
                        e_t = gamma_t.T @ cpf.Sigma_meas_inv @ gamma_t

                        time_steps.append(sim_time)
                        contact_detected.append(cpf.has_contact)
                        num_particles.append(result.shape[0])
                        residual_magnitudes.append(e_t)

                        if cpf.has_contact:
                            contact_positions.append(np.mean(result, axis=0))
                        else:
                            contact_positions.append(np.array([np.nan, np.nan, np.nan]))

                        step += 1

                        # Print progress every 100 steps
                        if step % 100 == 0:
                            print(f"Step: {step}/{num_steps}, Sim Time: {sim_time:.3f}")

                if current_time - last_update_time >= 1/60:  # Cap at 60 FPS
                    viewer.sync()
                    last_update_time = current_time

                # Sleep to maintain real-time simulation
                time_to_sleep = (start_time + step * step_duration) - time.time()
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)

        print("Visualization complete.")

        # Plot the results
        fig, axs = plt.subplots(4, 1, figsize=(12, 20))

        # Plot contact detection
        axs[0].plot(time_steps, contact_detected)
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Contact Detected')
        axs[0].set_title('Contact Detection over Time')
        axs[0].grid()

        # Plot number of particles
        axs[1].plot(time_steps, num_particles)
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Number of Particles')
        axs[1].set_title('Number of Particles over Time')
        axs[1].grid()

        # Plot contact positions
        contact_positions = np.array(contact_positions)
        axs[2].plot(time_steps, contact_positions[:, 0], label='X')
        axs[2].plot(time_steps, contact_positions[:, 1], label='Y')
        axs[2].plot(time_steps, contact_positions[:, 2], label='Z')
        axs[2].set_xlabel('Time (s)')
        axs[2].set_ylabel('Position')
        axs[2].set_title('Contact Position over Time')
        axs[2].legend()
        axs[2].grid()

        # Plot residual magnitudes
        axs[3].plot(time_steps, residual_magnitudes)
        axs[3].set_xlabel('Time (s)')
        axs[3].set_ylabel('Residual Magnitude')
        axs[3].set_title('Residual Magnitude over Time')
        axs[3].grid()

        plt.tight_layout()
        plt.show()

        return time_steps, contact_detected, num_particles, contact_positions, residual_magnitudes