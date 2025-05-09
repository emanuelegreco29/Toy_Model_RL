import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from env_sb3_dqn_moving import DiscretePointMassEnv

def load_latest_model(model_dir='models', prefix='dqn_sb3_moving_target_', ext='.zip'):
    files = glob.glob(os.path.join(model_dir, f"{prefix}*{ext}"))
    if not files:
        raise FileNotFoundError(f"No model files matching {prefix}*{ext}")
    latest = max(files, key=os.path.getmtime)
    print(f"Loading model: {latest}")
    return DQN.load(latest)

def evaluate_model(model, num_episodes=10):
    trajectories, target_trajs, rewards, distances = [], [], [], []
    env = Monitor(DiscretePointMassEnv(n_theta=3, n_z=3))
    for _ in range(num_episodes):
        obs, _ = env.reset()
        agent_path, target_path = [], []
        ep_reward = 0.0
        while True:
            agent_path.append(env.unwrapped.state[:3].copy())
            target_path.append(env.unwrapped.target_state[:3].copy())
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                agent_path.append(env.unwrapped.state[:3].copy())
                target_path.append(env.unwrapped.target_state[:3].copy())
                rewards.append(ep_reward)
                distances.append(info.get('distance', np.nan))
                break
        trajectories.append(np.array(agent_path))
        target_trajs.append(np.array(target_path))
    return trajectories, target_trajs, rewards, distances

def plot_separate_coordinates(agent_traj, target_traj, title_prefix='Episode', save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)
    coords = ['X', 'Y', 'Z']
    for i, label in enumerate(coords):
        plt.figure()
        plt.plot(agent_traj[:, i], label='Agent')
        plt.plot(target_traj[:, i], label='Target', linestyle='--')
        plt.xlabel('Timestep')
        plt.ylabel(f'{label}(t)')
        plt.title(f'{title_prefix} - {label}(t)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{title_prefix}_{label}.png'))
        plt.close()

def animate_trajectory(agent_traj, target_traj, filename='trajectory.gif'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    agent_line, = ax.plot([], [], [], lw=2, label='Agent')
    target_line, = ax.plot([], [], [], lw=2, linestyle='--', color='red', label='Target')

    # Set dynamic bounds to follow both agent and target trajectories
    all_points = np.vstack((agent_traj, target_traj))
    min_bounds = all_points.min(axis=0)
    max_bounds = all_points.max(axis=0)
    padding = 0.1 * (max_bounds - min_bounds)
    ax.set_xlim(min_bounds[0] - padding[0], max_bounds[0] + padding[0])
    ax.set_ylim(min_bounds[1] - padding[1], max_bounds[1] + padding[1])
    ax.set_zlim(min_bounds[2] - padding[2], max_bounds[2] + padding[2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    def update(num):
        agent_line.set_data(agent_traj[:num+1, 0], agent_traj[:num+1, 1])
        agent_line.set_3d_properties(agent_traj[:num+1, 2])
        target_line.set_data(target_traj[:num+1, 0], target_traj[:num+1, 1])
        target_line.set_3d_properties(target_traj[:num+1, 2])
        return agent_line, target_line

    ani = FuncAnimation(fig, update, frames=len(agent_traj), interval=200, blit=False)
    ani.save(filename, writer=PillowWriter(fps=5))
    plt.close()

# MAIN
model = load_latest_model()
num_ep = 100
trajectories, target_trajs, rewards, distances = evaluate_model(model, num_ep)

print(f"Average reward: {np.mean(rewards):.2f} | Average distance: {np.mean(distances):.2f}")

best_idx = int(np.argmax(rewards))
worst_idx = int(np.argmin(rewards))
print(f"Best Episode {best_idx+1} | Reward: {rewards[best_idx]:.2f} | Distance: {distances[best_idx]:.2f}")
print(f"Worst Episode {worst_idx+1} | Reward: {rewards[worst_idx]:.2f} | Distance: {distances[worst_idx]:.2f}")

# Plots x(t), y(t), z(t) for best and worst
plot_separate_coordinates(trajectories[best_idx], target_trajs[best_idx], title_prefix='Best_Episode')
plot_separate_coordinates(trajectories[worst_idx], target_trajs[worst_idx], title_prefix='Worst_Episode')

# Animated trajectory for best and worst
animate_trajectory(trajectories[best_idx], target_trajs[best_idx], filename='best_episode.gif')
animate_trajectory(trajectories[worst_idx], target_trajs[worst_idx], filename='worst_episode.gif')