import time
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

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

                    # Step the simulation
                    controller.step()

                # Update the viewer
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

        # Convert lists to numpy arrays for easier plotting
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
        residual_data = []

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

                    # Step the simulation
                    controller.step()

                    ctrl = controller.data.ctrl.copy()
                    qvel = controller.data.qvel.copy()
                    residual, _ = observer.get_residual(controller.get_time())
                   
                    time_steps.append(elapsed_time)
                    ctrl_data.append(ctrl)
                    qvel_data.append(qvel)
                    residual_data.append(residual)

                    print(f"Step: {step}")
                    # print(f"End effector position: {controller.get_end_effector_position()}")
                    # print(f"Control signals (data.ctrl): {ctrl}")
                    print(f"Control action: {controller.data.qfrc_applied}")
                    # print(f"Joint velocities (qvel): {qvel}")
                    print(f"Residual: {residual}")
                    print("---")
                    
                    step += 1

                # Update the viewer
                if current_time - last_update_time >= 1/60:  # Cap at 60 FPS
                    viewer.sync()
                    last_update_time = current_time

                # Sleep to maintain real-time simulation
                time_to_sleep = (start_time + step * step_duration) - time.time()
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)

        print("Visualization complete.")
