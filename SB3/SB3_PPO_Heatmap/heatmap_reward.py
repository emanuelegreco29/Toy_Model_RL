import numpy as np
import matplotlib.pyplot as plt

target_pos = np.array([0,0,0])
target_dir = np.array([1,0,0])

def compute_reward_pointwise(agent_pos, target_pos, target_dir, yaw=0, pitch=0):
        vec = agent_pos - target_pos
        d3 = np.linalg.norm(vec) + 1e-8
        f_dist = np.exp(-0.5 * (d3/10) ** 1) 

        ux, uy, uz = vec / d3
        position_vec = np.array([ux, uy, uz])
        # kappa = 1
        # alignment_pos = (1 + (ux*target_dir[0] + uy*target_dir[1] + uz*target_dir[2])**kappa) / 2
        # f_head_pos = np.clip(alignment_pos, 0, 1)
        
        dx = np.cos(pitch) * np.cos(yaw)
        dy = np.cos(pitch) * np.sin(yaw)
        dz = np.sin(pitch)
        direction_vec = np.array([dx, dy, dz])
        # alignment_vel = (1 + (dx*target_dir[0] + dy*target_dir[1] + dz*target_dir[2])**kappa) / 2
        # f_head_vel = np.clip(alignment_vel, 0, 1)
        
        # Compute the dot product and normalize
        cos_vel = np.clip(np.dot(direction_vec, target_dir), -1.0, 1.0)
        cos_pos = np.clip(np.dot(position_vec, target_dir), -1.0, 1.0)
        
        f_head_pos = (1 - cos_pos) / 2
        f_head_vel = (1 + cos_vel) / 2

        return (f_dist * 0.5 + f_head_pos * 0.3 + f_head_vel * 0.2) - 1.0

#create a 3D grid of points in the space [-3,3]x[-3,3]x[-3,3]
def create_grid_points(min_val=-30, max_val=30, step=0.1):
    x = np.arange(min_val, max_val + step, step)
    y = np.arange(min_val, max_val + step, step)
    z = np.array([0])  # z=0 for a 2D heatmap
    return np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

# Create the grid points
grid_points = create_grid_points()

# compute the reward for each point in the grid
rewards = []
for point in grid_points:
    reward = compute_reward_pointwise(point, target_pos, target_dir)
    rewards.append(reward)
# Convert rewards to a numpy array
rewards = np.array(rewards)

#plot a 2D heatmap of the rewards in the x-y plane at z=0
def plot_heatmap(rewards, grid_points):
    # Filter points at z=0

    # Create a 2D grid for plotting
    x_unique = np.unique(grid_points[:, 0])
    y_unique = np.unique(grid_points[:, 1])
    Z = rewards.reshape(len(y_unique), len(x_unique)).T

    plt.figure(figsize=(8, 6))
    plt.imshow(Z, extent=(x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()), origin='lower', cmap='viridis')
    plt.colorbar(label='Reward')
    plt.title('Reward Heatmap at z=0')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.show()
# Plot the heatmap
plot_heatmap(rewards, grid_points)