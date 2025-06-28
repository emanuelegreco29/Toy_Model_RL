import os, glob, random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from stable_baselines3 import PPO
import datetime

from env_sb3_ppo_multi import PointMassEnv


def load_latest_model(model_dir='models', prefix='ppo_sb3_multi_', ext='.zip'):
    files = glob.glob(os.path.join(model_dir, f"{prefix}*{ext}"))
    if not files:
        raise FileNotFoundError(f"No model files matching {prefix}*{ext}")
    latest = max(files, key=os.path.getmtime)
    print(f"Loading model: {latest}")
    return PPO.load(latest, device='cpu')


def evaluate_model(model, num_episodes=10):
    trajectories, target_trajs, rewards, distances, followed = [], [], [], [], []
    env = PointMassEnv()
    for ep in range(num_episodes):
        obs, _ = env.reset()
        agent_path, target_path = [], []
        ep_reward = 0.0
        while True:
            agent_path.append(env.state[:3].copy())
            target_path.append(env.target_state[:3].copy())
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                agent_path.append(env.state[:3].copy())
                target_path.append(env.target_state[:3].copy())
                rewards.append(ep_reward)
                distances.append(info.get('distance', np.nan))
                followed.append(info.get('followed', np.nan))
                break
        trajectories.append(np.array(agent_path))
        target_trajs.append(np.array(target_path))
    return trajectories, target_trajs, rewards, distances, followed


def plot_coordinates(agent_traj, target_traj, title_prefix='Episode', save_dir='plots'):
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

def plot_velocities(agent_traj, target_traj, title_prefix='Episode', save_dir='plots',
                    vmin=0.1, vmax=1.5, dt=0.1):
    """
    Calcola la velocità scalare come norma del vettore velocità (displacement/dt),
    e la plotta per agente e target, con limiti y su [vmin, vmax].
    """
    os.makedirs(save_dir, exist_ok=True)
    # displacement tra posizioni consecutive
    agent_disp = np.diff(agent_traj, axis=0) / dt
    target_disp = np.diff(target_traj, axis=0) / dt
    # velocità scalari
    agent_vel = np.linalg.norm(agent_disp, axis=1)
    target_vel = np.linalg.norm(target_disp, axis=1)
    timesteps = np.arange(1, len(agent_vel) + 1)

    plt.figure()
    plt.plot(timesteps, agent_vel, label='Agent speed')
    plt.plot(timesteps, target_vel, label='Target speed', linestyle='--')
    # linee di riferimento
    plt.axhline(vmin, color='gray', linestyle=':', label='vmin')
    plt.axhline(vmax, color='gray', linestyle='-.', label='vmax')
    plt.xlabel('Timestep')
    plt.ylabel('Speed')
    plt.title(f'{title_prefix} – Speed Variation')
    plt.ylim(vmin * 0.9, vmax * 1.1)  # piccolo margine intorno a [vmin, vmax]
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{title_prefix}_speed.png'))
    plt.close()

def animate_trajectory(agent_traj, target_traj, filename='trajectory.gif', save_dir='plots', pad=1.0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    agent_line, = ax.plot([], [], [], lw=2, color='blue', label='Agent')
    target_line, = ax.plot([], [], [], lw=2, linestyle='-', color='orange', label='Target')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    def update(num):
        agent_line.set_data(agent_traj[:num+1, 0], agent_traj[:num+1, 1])
        agent_line.set_3d_properties(agent_traj[:num+1, 2])
        target_line.set_data(target_traj[:num+1, 0], target_traj[:num+1, 1])
        target_line.set_3d_properties(target_traj[:num+1, 2])

        curr_agent = agent_traj[num]
        curr_target = target_traj[num]
        mins = np.minimum(curr_agent, curr_target) - pad
        maxs = np.maximum(curr_agent, curr_target) + pad
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])

        return agent_line, target_line

    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    ani = FuncAnimation(fig, update, frames=len(agent_traj), interval=200, blit=False)
    ani.save(filepath, writer=PillowWriter(fps=5))
    plt.close()


model = load_latest_model()
print("Model loaded successfully.")
timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M')
save_dir = os.path.join('plots', f"sb3_multi_eval_{timestamp}")
os.makedirs(save_dir, exist_ok=True)
num_ep = 100
print(f"Evaluating model for {num_ep} episodes...")
trajectories, target_trajs, rewards, distances, followed = evaluate_model(model, num_ep)

print(f"Average reward: {np.mean(rewards):.2f} | Average following steps: {np.mean(followed):.2f}")

best_idx = int(np.argmax(rewards))
worst_idx = int(np.argmin(rewards))
print(f"Best Episode {best_idx+1} | Reward: {rewards[best_idx]:.2f} | Followed target for: {followed[best_idx]}")
print(f"Worst Episode {worst_idx+1} | Reward: {rewards[worst_idx]:.2f} | Followed target for: {followed[worst_idx]}")

print("Plotting coordinate time-series...")
plot_coordinates(trajectories[best_idx], target_trajs[best_idx], title_prefix='Best_Episode', save_dir=save_dir)
plot_coordinates(trajectories[worst_idx], target_trajs[worst_idx], title_prefix='Worst_Episode', save_dir=save_dir)

print("Plotting speed time-series...")
plot_velocities(trajectories[best_idx], target_trajs[best_idx], title_prefix='Best_Episode', save_dir=save_dir)
plot_velocities(trajectories[worst_idx], target_trajs[worst_idx], title_prefix='Worst_Episode', save_dir=save_dir)

print("Creating animations...")
animate_trajectory(trajectories[best_idx], target_trajs[best_idx], filename='best_episode.gif', save_dir=save_dir)
animate_trajectory(trajectories[worst_idx], target_trajs[worst_idx], filename='worst_episode.gif', save_dir=save_dir)
print(f"Plots and animations saved in {save_dir}")
