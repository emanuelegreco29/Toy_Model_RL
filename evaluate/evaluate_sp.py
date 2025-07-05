import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D

from environments.dogfight_env import DogfightParallelEnv
from algorithms.PPO.custom_ppo import CustomActorCritic

def load_latest_policies(base_dir='policies/SP'):
    folders = glob.glob(os.path.join(base_dir, 'SelfPlay_*'))
    if not folders:
        raise FileNotFoundError(f"No SelfPlay folders found in {base_dir}")
    latest_folder = max(folders, key=os.path.getmtime)
    print(f"Loading policies from: {latest_folder}")
    ch_files = glob.glob(os.path.join(latest_folder, 'selfplay_chaser_ep*.pth'))
    ev_files = glob.glob(os.path.join(latest_folder, 'selfplay_evader_ep*.pth'))
    if not ch_files or not ev_files:
        raise FileNotFoundError("Policy files not found in the latest SelfPlay folder")
    return max(ch_files, key=os.path.getmtime), max(ev_files, key=os.path.getmtime)

def evaluate_policies(chaser_net, evader_net, env, num_episodes=100):
    trajectories, target_trajs, rewards, final_hps, hit_list = [], [], [], [], []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        prev_hp = env.hp['evader_0']
        ch_traj, ev_traj, hits = [], [], []
        total_reward = 0.0
        step = 0
        while not done:
            ch_traj.append(env.states['chaser_0'].copy())
            ev_traj.append(env.states['evader_0'].copy())
            actions = {}
            for name, net in [('chaser_0', chaser_net), ('evader_0', evader_net)]:
                tensor_obs = torch.tensor(obs[name], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    mean, log_std, _ = net(tensor_obs)
                actions[name] = mean.squeeze().numpy()
            next_obs, rews, dones, _ = env.step(actions)
            total_reward += rews['chaser_0']
            curr_hp = env.hp['evader_0']
            if curr_hp < prev_hp:
                hits.append(step)
            prev_hp = curr_hp
            obs = next_obs
            done = dones['chaser_0']
            step += 1
        trajectories.append(np.array(ch_traj))
        target_trajs.append(np.array(ev_traj))
        rewards.append(total_reward)
        final_hps.append(env.hp['evader_0'])
        hit_list.append(hits)
    return trajectories, target_trajs, rewards, final_hps, hit_list

def plot_coordinates(agent_traj, target_traj, hit_indices, title_prefix, save_dir):
    coords = ['X', 'Y', 'Z']
    for i, label in enumerate(coords):
        plt.figure()
        plt.plot(agent_traj[:, i], label='Chaser')
        plt.plot(target_traj[:, i], label='Evader', linestyle='--')
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
    # assumiamo che DogfightParallelEnv esponga wez_length e wez_angle
    env_params = DogfightParallelEnv()
    wez_length = getattr(env_params, 'wez_length', getattr(env_params, 'wez_radius', 1000.0))
    wez_angle = getattr(env_params, 'wez_angle', np.pi/6)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    agent_line, = ax.plot([], [], [], lw=2, label='Chaser')
    target_line, = ax.plot([], [], [], lw=2, linestyle='--', label='Evader')
    boundary1, = ax.plot([], [], [], lw=1, alpha=0.7, label='WEZ Edge')
    boundary2, = ax.plot([], [], [], lw=1, alpha=0.7)

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()

    def update(num):
        agent = agent_traj[num]
        target = target_traj[num]
        # traiettorie
        agent_line.set_data(agent_traj[:num+1,0], agent_traj[:num+1,1])
        agent_line.set_3d_properties(agent_traj[:num+1,2])
        target_line.set_data(target_traj[:num+1,0], target_traj[:num+1,1])
        target_line.set_3d_properties(target_traj[:num+1,2])

        # margini WEZ a wedge
        yaw = agent[4]
        pitch = agent[5]
        left = np.array([
            np.cos(pitch)*np.cos(yaw + wez_angle),
            np.cos(pitch)*np.sin(yaw + wez_angle),
            np.sin(pitch)
        ]) * wez_length
        right = np.array([
            np.cos(pitch)*np.cos(yaw - wez_angle),
            np.cos(pitch)*np.sin(yaw - wez_angle),
            np.sin(pitch)
        ]) * wez_length
        p1 = agent[:3] + left
        p2 = agent[:3] + right

        boundary1.set_data([agent[0], p1[0]], [agent[1], p1[1]])
        boundary1.set_3d_properties([agent[2], p1[2]])
        boundary2.set_data([agent[0], p2[0]], [agent[1], p2[1]])
        boundary2.set_3d_properties([agent[2], p2[2]])

        # limiti dinamici
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

# === Main Evaluation Script ===

# Caricamento policy
chaser_path, evader_path = load_latest_policies()
env = DogfightParallelEnv(K_history=1)
obs_dim = env.observation_spaces['chaser_0'].shape[0]
act_dim = env.action_spaces['chaser_0'].shape[0]
chaser_net = CustomActorCritic(obs_dim, act_dim)
chaser_net.load_state_dict(torch.load(chaser_path, map_location='cpu'))
for p in chaser_net.parameters(): p.requires_grad=False
evader_net = CustomActorCritic(obs_dim, act_dim)
evader_net.load_state_dict(torch.load(evader_path, map_location='cpu'))
for p in evader_net.parameters(): p.requires_grad=False

# Cartelle di output
timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M')
save_dir = os.path.join('plots', f'eval_selfplay_{timestamp}')
os.makedirs(save_dir, exist_ok=True)

# Valutazione
print('Evaluating policies for 100 episodes...')
trajectories, target_trajs, rewards, final_hps, hits = evaluate_policies(
    chaser_net, evader_net, env, num_episodes=100
)

# Selezione best/worst
min_hp, max_hp = min(final_hps), max(final_hps)
best_idx = max([i for i,hp in enumerate(final_hps) if hp==min_hp], key=lambda i: rewards[i])
worst_idx = min([i for i,hp in enumerate(final_hps) if hp==max_hp], key=lambda i: rewards[i])
print(f'Best Episode {best_idx+1} | Final HP: {final_hps[best_idx]} | Reward: {rewards[best_idx]:.2f}')
print(f'Worst Episode {worst_idx+1} | Final HP: {final_hps[worst_idx]} | Reward: {rewards[worst_idx]:.2f}')

# Plot coordinate nel tempo
print("Plotting coordinate time-series with hit points...")
plot_coordinates(trajectories[best_idx], target_trajs[best_idx], hits[best_idx], 'Best_Episode', save_dir)
plot_coordinates(trajectories[worst_idx], target_trajs[worst_idx], hits[worst_idx], 'Worst_Episode', save_dir)

# Animazioni WEZ
print("Creating animations with WEZ...")
animate_with_wez(trajectories[best_idx], target_trajs[best_idx], 'best_episode.gif', save_dir)
animate_with_wez(trajectories[worst_idx], target_trajs[worst_idx], 'worst_episode.gif', save_dir)

print(f'Plots and animations saved in {save_dir}')