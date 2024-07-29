import time
import mujoco.viewer

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
                    print(f"End effector position: {controller.get_end_effector_position()}")
                    print(f"Control signals (data.ctrl): {ctrl}")
                    print(f"Joint velocities (qvel): {qvel}")
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
