import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation
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
                with torch.no_grad(): mean, log_std, _ = net(tensor_obs)
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


def plot_3d(agent_traj, target_traj, hit_indices, prefix, save_dir):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plot only XYZ coords
    ax.plot(agent_traj[:,0], agent_traj[:,1], agent_traj[:,2], 'b-', label='Chaser')
    ax.plot(target_traj[:,0], target_traj[:,1], target_traj[:,2], 'r--', label='Evader')
    if hit_indices:
        pts = target_traj[hit_indices, :3]
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], c='k', marker='x', label='Hit')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(f'{prefix} - 3D Trajectory')
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_3d.png'))
    plt.close()


def animate_3d(agent_traj, target_traj, hit_indices, wez_radius, prefix, save_dir):
    # extract only XYZ coords
    a_xyz = agent_traj[:, :3]
    e_xyz = target_traj[:, :3]
    # prepare sphere offsets for WEZ
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    sphere_offsets = np.stack((np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)), axis=-1)
    sphere_offsets = sphere_offsets.reshape(-1,3) * wez_radius

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # set limits based on positions
    all_pts = np.vstack((a_xyz, e_xyz))
    mins, maxs = all_pts.min(axis=0), all_pts.max(axis=0)
    ax.set_xlim(mins[0]-wez_radius, maxs[0]+wez_radius)
    ax.set_ylim(mins[1]-wez_radius, maxs[1]+wez_radius)
    ax.set_zlim(mins[2]-wez_radius, maxs[2]+wez_radius)

    line_c, = ax.plot([], [], [], 'b-', label='Chaser')
    line_e, = ax.plot([], [], [], 'r--', label='Evader')
    scatter_h, = ax.plot([], [], [], marker='x', linestyle='None', color='k', label='Hit')
    scatter_w = ax.scatter([], [], [], c='gray', alpha=0.3, s=1, label='WEZ')
    ax.set_title(f'{prefix} - 3D Animation'); ax.legend()

    def init():
        line_c.set_data([], []); line_c.set_3d_properties([])
        line_e.set_data([], []); line_e.set_3d_properties([])
        scatter_h.set_data([], []); scatter_h.set_3d_properties([])
        scatter_w._offsets3d = ([], [], [])
        return line_c, line_e, scatter_h, scatter_w

    def update(frame):
        # update trajectories
        line_c.set_data(a_xyz[:frame,0], a_xyz[:frame,1])
        line_c.set_3d_properties(a_xyz[:frame,2])
        line_e.set_data(e_xyz[:frame,0], e_xyz[:frame,1])
        line_e.set_3d_properties(e_xyz[:frame,2])
        # update hits
        curr_hits = [h for h in hit_indices if h < frame]
        if curr_hits:
            pts = e_xyz[curr_hits]
            scatter_h.set_data(pts[:,0], pts[:,1]); scatter_h.set_3d_properties(pts[:,2])
        # update WEZ sphere
        center = e_xyz[frame]
        sph = sphere_offsets + center
        scatter_w._offsets3d = (sph[:,0], sph[:,1], sph[:,2])
        return line_c, line_e, scatter_h, scatter_w

    anim = FuncAnimation(fig, update, init_func=init, frames=len(a_xyz), blit=True)
    out_path = os.path.join(save_dir, f'{prefix}_3d_anim.mp4')
    anim.save(out_path, fps=30, dpi=200)
    plt.close()

# === Main Evaluation Script ===
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

timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M')
save_dir = os.path.join('plots', f'eval_selfplay_{timestamp}')
os.makedirs(save_dir, exist_ok=True)

print('Evaluating policies for 100 episodes...')
trajectories, target_trajs, rewards, final_hps, hit_indices_list = evaluate_policies(
    chaser_net, evader_net, env, num_episodes=100
)

min_hp, max_hp = min(final_hps), max(final_hps)
best_idx = max([i for i,hp in enumerate(final_hps) if hp==min_hp], key=lambda i: rewards[i])
worst_idx = min([i for i,hp in enumerate(final_hps) if hp==max_hp], key=lambda i: rewards[i])

print(f'Best Episode {best_idx+1} | Final HP: {final_hps[best_idx]} | Reward: {rewards[best_idx]:.2f}')
print(f'Worst Episode {worst_idx+1} | Final HP: {final_hps[worst_idx]} | Reward: {rewards[worst_idx]:.2f}')

# Static 3D
plot_3d(trajectories[best_idx], target_trajs[best_idx], hit_indices_list[best_idx], f'Best_{best_idx+1}', save_dir)
plot_3d(trajectories[worst_idx], target_trajs[worst_idx], hit_indices_list[worst_idx], f'Worst_{worst_idx+1}', save_dir)

# Animate 3D with hits and WEZ
wez_radius = getattr(env, 'wez_radius', 1000.0)
print('Animating 3D trajectories with WEZ...')
animate_3d(trajectories[best_idx], target_trajs[best_idx], hit_indices_list[best_idx], wez_radius, f'Best_{best_idx+1}', save_dir)
animate_3d(trajectories[worst_idx], target_trajs[worst_idx], hit_indices_list[worst_idx], wez_radius, f'Worst_{worst_idx+1}', save_dir)

print(f'All 3D plots and animations saved to {save_dir}')
