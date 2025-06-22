import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from env_sb3_ppo_heatmap import PointMassEnv

# Create environment and advance to a chosen step
env = PointMassEnv()
obs, _ = env.reset()
# Advance a few steps to move target
for _ in range(10):
    obs, reward, done, truncated, info = env.step(env.action_space.sample())

# Get target position and direction from history
target_pos = env.target_state
prev_t = env.target_history[-2]
delta = target_pos - prev_t
target_dir = delta / (np.linalg.norm(delta) + 1e-8)

# Define sampling grid around target
grid_size = 100  # number of points per axis
dist_range = 10.0  # meters around target
xs = np.linspace(target_pos[0] - dist_range, target_pos[0] + dist_range, grid_size)
ys = np.linspace(target_pos[1] - dist_range, target_pos[1] + dist_range, grid_size)
zs = np.linspace(target_pos[2] - dist_range, target_pos[2] + dist_range, grid_size)
X, Y, Z = np.meshgrid(xs, ys, zs, indexing='xy')

# Compute reward field pointwise
rewards = np.zeros_like(X)
for i in range(grid_size):
    for j in range(grid_size):
        for k in range(grid_size):
            pos = np.array([X[i,j,k], Y[i,j,k], Z[i,j,k]])
            rewards[i,j,k] = env.compute_reward_pointwise(pos, target_pos, target_dir)

# Flatten for scatter plot
x_flat = X.flatten()
y_flat = Y.flatten()
z_flat = Z.flatten()
v_flat = rewards.flatten()

# Filter low rewards for clarity
thr = 0.2
mask = v_flat > thr

# Plot 3D scatter and target
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
# Reward field points
sc = ax.scatter(x_flat[mask], y_flat[mask], z_flat[mask], c=v_flat[mask], cmap='viridis',
                marker='o', s=20, alpha=0.6)
# Target position marker
ax.scatter([target_pos[0]], [target_pos[1]], [target_pos[2]], c='red', s=100, marker='o', label='Target')
fig.colorbar(sc, ax=ax, label='Reward')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Reward Field Using compute_reward_pointwise')
ax.legend()
plt.tight_layout()
plt.show()