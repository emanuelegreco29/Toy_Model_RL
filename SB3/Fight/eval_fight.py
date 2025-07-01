import os, glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from stable_baselines3 import PPO
import datetime
import torch

from env_fight import PointMassEnv

def load_latest_model(model_dir='models', prefix='ppo_fight_', ext='.zip'):
    files = glob.glob(os.path.join(model_dir, f"{prefix}*{ext}"))
    if not files:
        raise FileNotFoundError(f"No model files matching {prefix}*{ext}")
    latest = max(files, key=os.path.getmtime)
    print(f"Loading model: {latest}")
    return PPO.load(latest, device='cpu')


def evaluate_model(model, num_episodes=10):
    trajectories, target_trajs, rewards, final_hps, hit_indices_list = [], [], [], [], []
    env = PointMassEnv()
    for ep in range(num_episodes):
        obs, _ = env.reset()
        agent_path, target_path = [], []
        ep_reward = 0.0
        hp_prev = env.target_hp
        hit_indices = []
        step = 0
        while True:
            # record full state for WEZ orientation
            agent_path.append(env.state.copy())
            target_path.append(env.target_state.copy())

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            # detect hits by HP drop
            hp_curr = env.target_hp
            if hp_curr < hp_prev:
                hit_indices.append(step)
            hp_prev = hp_curr
            step += 1

            if terminated or truncated:
                # append final state
                agent_path.append(env.state.copy())
                target_path.append(env.target_state.copy())
                trajectories.append(np.array(agent_path))
                target_trajs.append(np.array(target_path))
                rewards.append(ep_reward)
                final_hps.append(info.get('final_hp', np.nan))
                hit_indices_list.append(hit_indices)
                break

    return trajectories, target_trajs, rewards, final_hps, hit_indices_list


def plot_coordinates(agent_traj, target_traj, hit_indices, title_prefix, save_dir):
    coords = ['X', 'Y', 'Z']
    for i, label in enumerate(coords):
        plt.figure()
        plt.plot(agent_traj[:, i], label='Agent')
        plt.plot(target_traj[:, i], label='Target', linestyle='--')
        if hit_indices:
            plt.scatter(hit_indices, target_traj[hit_indices, i], marker='x', label='Hit')
        plt.xlabel('Timestep')
        plt.ylabel(f'{label}(t)')
        plt.title(f'{title_prefix} - {label}(t)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{title_prefix}_{label}.png'))
        plt.close()


def animate_with_wez(agent_traj, target_traj, filename, save_dir, pad=1.0):
    env_params = PointMassEnv()
    wez_length = env_params.wez_length
    wez_angle = env_params.wez_angle

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    agent_line, = ax.plot([], [], [], lw=2, color='blue', label='Agent')
    target_line, = ax.plot([], [], [], lw=2, linestyle='-', color='orange', label='Target')
    boundary1, = ax.plot([], [], [], lw=1, color='green', alpha=0.5, label='WEZ Edge')
    boundary2, = ax.plot([], [], [], lw=1, color='green', alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    def update(num):
        agent = agent_traj[num]
        target = target_traj[num]
        # update agent and target trajectories
        agent_line.set_data(agent_traj[:num+1, 0], agent_traj[:num+1, 1])
        agent_line.set_3d_properties(agent_traj[:num+1, 2])
        target_line.set_data(target_traj[:num+1, 0], target_traj[:num+1, 1])
        target_line.set_3d_properties(target_traj[:num+1, 2])

        # compute WEZ edges based on current yaw/pitch
        yaw = agent[4]
        pitch = agent[5]
        # central direction (not plotted) and edges
        left = np.array([np.cos(pitch)*np.cos(yaw + wez_angle),
                         np.cos(pitch)*np.sin(yaw + wez_angle),
                         np.sin(pitch)])
        right = np.array([np.cos(pitch)*np.cos(yaw - wez_angle),
                          np.cos(pitch)*np.sin(yaw - wez_angle),
                          np.sin(pitch)])
        p1 = agent[:3] + left * wez_length
        p2 = agent[:3] + right * wez_length

        boundary1.set_data([agent[0], p1[0]], [agent[1], p1[1]])
        boundary1.set_3d_properties([agent[2], p1[2]])
        boundary2.set_data([agent[0], p2[0]], [agent[1], p2[1]])
        boundary2.set_3d_properties([agent[2], p2[2]])

        # adjust axes limits
        mins = np.minimum(agent[:3], target[:3]) - pad
        maxs = np.maximum(agent[:3], target[:3]) + pad
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])

        return agent_line, target_line, boundary1, boundary2

    filepath = os.path.join(save_dir, filename)
    ani = FuncAnimation(fig, update, frames=len(agent_traj), interval=200, blit=False)
    ani.save(filepath, writer=PillowWriter(fps=5))
    plt.close()



model = load_latest_model()
print("Model loaded successfully.")
log_std = model.policy.log_std.detach().cpu().numpy()
std = torch.exp(model.policy.log_std).detach().cpu().numpy()
print("Log-std per dimensione d’azione:", log_std)
print("Std per dimensione d’azione:   ", std)
timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M')
save_dir = os.path.join('plots', f"fight_eval_{timestamp}")
os.makedirs(save_dir, exist_ok=True)

num_ep = 100
print(f"Evaluating model for {num_ep} episodes...")
trajectories, target_trajs, rewards, final_hps, hits = evaluate_model(model, num_ep)

# select best/worst by final HP, tie-breaking on reward
min_hp = np.min(final_hps)
max_hp = np.max(final_hps)
best_idxs = [i for i, hp in enumerate(final_hps) if hp == min_hp]
worst_idxs = [i for i, hp in enumerate(final_hps) if hp == max_hp]
best_idx = max(best_idxs, key=lambda i: rewards[i])
worst_idx = min(worst_idxs, key=lambda i: rewards[i])

print(f"Best Episode {best_idx+1} | Final HP: {final_hps[best_idx]} | Reward: {rewards[best_idx]:.2f}")
print(f"Worst Episode {worst_idx+1} | Final HP: {final_hps[worst_idx]} | Reward: {rewards[worst_idx]:.2f}")

print("Plotting coordinate time-series with hit points...")
plot_coordinates(trajectories[best_idx], target_trajs[best_idx], hits[best_idx], 'Best_Episode', save_dir)
plot_coordinates(trajectories[worst_idx], target_trajs[worst_idx], hits[worst_idx], 'Worst_Episode', save_dir)

print("Creating animations with WEZ...")
animate_with_wez(trajectories[best_idx], target_trajs[best_idx], 'best_episode.gif', save_dir)
animate_with_wez(trajectories[worst_idx], target_trajs[worst_idx], 'worst_episode.gif', save_dir)

print(f"Plots and animations saved in {save_dir}")
