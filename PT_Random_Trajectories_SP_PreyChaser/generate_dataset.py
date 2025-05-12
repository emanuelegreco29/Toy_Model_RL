import numpy as np
import env_PT_rand_traj

# Parameters from your environments
dt = env_PT_rand_traj.PointMassEnv.dt
T  = env_PT_rand_traj.PointMassEnv.max_steps
initial_speed = 1.0
max_step = initial_speed * dt

# Default sampling bounds
bounds = np.array([
    [0, +10.0],   # x
    [0, +10.0],   # y
    [0, +10.0],   # z
], dtype=np.float32)

def generate_dataset(n_trajectories=100000, output_path="dataset.npz", K=6):
    states = []
    next_positions = []

    for _ in range(n_trajectories):
        traj = env_PT_rand_traj.generate_bspline_trajectory(
            bounds, max_step, dt, T=T, K=K
        )  # shape (T,3)

        for i in range(T - 1):
            x, y, z = traj[i]
            dx, dy, dz = traj[i+1] - traj[i]
            theta = np.arctan2(dy, dx)
            v = np.linalg.norm([dx, dy, dz]) / dt
            states.append([x, y, z, v, theta])
            next_positions.append(traj[i+1])

    states = np.array(states, dtype=np.float32)
    next_positions = np.array(next_positions, dtype=np.float32)
    np.savez_compressed(output_path, states=states, next_positions=next_positions)
    print(f"Saved dataset with {states.shape[0]} samples to {output_path}")

NUM = 50000
PATH = f"dataset_{NUM}.npz"
K = 20

generate_dataset(n_trajectories=NUM, output_path=PATH, K=K)