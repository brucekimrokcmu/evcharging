import os
import numpy as np
from dm_control import mujoco
from contact_particle_filter import ContactParticleFilter
from residual_observer import ResidualObserver


def main():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(curr_dir, "../data/universal_robots_ur10e/scene.xml")
    config_path = os.path.join(curr_dir, "config.json")

    physics = mujoco.Physics.from_xml_path(model_path)

    cpf = ContactParticleFilter(physics, config_path)

    test_sample_particles(cpf)
    test_frame_conversions(cpf)
    test_motion_model(cpf)
    test_measurement_model(cpf)
    test_run_contact_particle_filter(cpf)

def test_sample_particles(cpf):
    print("Testing sample_particles_on_meshes()...")
    cpf.sample_particles_on_meshes()
    print(f"Number of particles: {cpf.particles_mesh_frame.shape[0]}")
    print(f"Sample particle positions:\n{cpf.particles_mesh_frame[:5]}")
    print(f"Sample particle indices:\n{cpf.indices_mesh_frame[:5]}")

def test_frame_conversions(cpf):
    print("\nTesting frame conversions...")
    original_mesh_frame = cpf.particles_mesh_frame.copy()
    world_frame = cpf._from_mesh_to_world_frame()
    print(f"World frame sample:\n{world_frame[:5]}")
    
    mesh_frame = cpf._from_world_to_mesh_frame(world_frame)
    print(f"Back to mesh frame sample:\n{mesh_frame[:5]}")
    assert np.allclose(original_mesh_frame, mesh_frame, atol=1e-5), "Mesh to World to Mesh conversion error"

    # Convert the world frame back to world (should be identity operation)
    world_frame_2 = cpf._from_mesh_to_world_frame()
    assert np.allclose(world_frame, world_frame_2, atol=1e-5), "World to Mesh to World conversion error"
    
    print("Frame conversions test passed successfully!")

def test_motion_model(cpf):
    print("\nTesting run_motion_model()...")
    initial_particles = cpf.particles_mesh_frame.copy()
    cpf.run_motion_model()
    print(f"Motion model displacement sample:\n{cpf.particles_mesh_frame[:5] - initial_particles[:5]}")

def test_measurement_model(cpf):
    print("\nTesting run_measurement_model()...")
    gamma_t, _ = cpf.residual.get_residual(0)  # Assuming time=0
    particles_world_frame = cpf._from_mesh_to_world_frame()
    normalized_Xt_bar = cpf.run_measurement_model(gamma_t, particles_world_frame)
    print(f"Measurement model output sample:\n{normalized_Xt_bar[:5]}")

def test_run_contact_particle_filter(cpf):
    print("\nTesting run_contact_particle_filter()...")
    for t in range(100):  # Run for 10 time steps
        result = cpf.run_contact_particle_filter(t)
        gamma_t, _ = cpf.residual.get_residual(t)
        e_t = gamma_t.T @ cpf.Sigma_meas_inv @ gamma_t
        print(f"Time step {t}:")
        print(f"  Contact detected: {cpf.has_contact}")
        print(f"  Residual magnitude: {e_t}")
        print(f"  Number of particles: {result.shape[0]}")
        if cpf.has_contact:
            print(f"  Sample particle positions:\n{result[:5]}")

if __name__ == "__main__":
    main()