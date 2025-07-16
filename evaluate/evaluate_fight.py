import os
import sys
import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environments.fight_env import FightEnv
from algorithms.PPO.custom_ppo import CustomActorCritic

def load_latest_policies(base_dir='policies/Fight_SP'):
    """
    Cerca la cartella Fight_SelfPlay più recente in base_dir,
    e restituisce i due file di policy più recenti per Agent 0 e Agent 1.
    """
    folders = glob.glob(os.path.join(base_dir, 'Fight_SelfPlay_*'))
    if not folders:
        raise FileNotFoundError(f"No Fight_SelfPlay folders found in {base_dir}")
    latest = max(folders, key=os.path.getmtime)
    print(f"Loading policies from: {latest}")
    files = glob.glob(os.path.join(latest, '*_policy_ep*.pth'))
    chaser_file = max([f for f in files if 'Agent 0' in f], key=os.path.getmtime)
    evader_file = max([f for f in files if 'Agent 1' in f], key=os.path.getmtime)
    return chaser_file, evader_file

def evaluate_policies(chaser_net, evader_net, env, num_episodes=100):
    """
    Esegue num_episodes di roll-out self-play e raccoglie
    le traiettorie e il conteggio 'followed' per ciascuna epoca.
    """
    trajs = []
    followed = []
    for ep in range(num_episodes):
        obs, infos = env.reset()
        done = False
        ch_traj, ev_traj = [], []
        while not done:
            # salva posizione corrente
            ch_traj.append(env.states['Agent 0'][:3].copy())
            ev_traj.append(env.states['Agent 1'][:3].copy())
            # scelta action: prendo la media (no sampling)
            actions = {}
            for name, net in [('Agent 0', chaser_net), ('Agent 1', evader_net)]:
                o = torch.tensor(obs[name], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    mean, log_std, _ = net(o)
                actions[name] = mean.squeeze().numpy()
            obs, rews, dones, infos = env.step(actions)
            done = any(dones.values())
        trajs.append((np.array(ch_traj), np.array(ev_traj)))
        f0 = infos['Agent 0']['followed']
        f1 = infos['Agent 1']['followed']
        followed.append((f0, f1))
    return trajs, followed

def plot_trajectory(ch_traj, ev_traj, title, save_dir):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(ch_traj[:,0], ch_traj[:,1], ch_traj[:,2], label='Agent 0 (Chaser)')
    ax.plot(ev_traj[:,0], ev_traj[:,1], ev_traj[:,2],
            linestyle='--', label='Agent 1 (Evader)')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, f"{title}.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved trajectory plot: {path}")

def set_axes_equal(ax):
    """Imposta scala uguale sugli assi 3D."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    max_range = max(x_range, y_range, z_range) / 2.0
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)
    ax.set_xlim3d([x_middle - max_range, x_middle + max_range])
    ax.set_ylim3d([y_middle - max_range, y_middle + max_range])
    ax.set_zlim3d([z_middle - max_range, z_middle + max_range])

def animate_trajectory(ch_traj, ev_traj, filename, save_dir):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=45)
    ch_line, = ax.plot([], [], [], lw=2, label='Agent 0 (Chaser)')
    ev_line, = ax.plot([], [], [], lw=2, linestyle='--', label='Agent 1 (Evader)')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()
    margin = 0.5

    def update(frame):
        ch_line.set_data(ch_traj[:frame+1,0], ch_traj[:frame+1,1])
        ch_line.set_3d_properties(ch_traj[:frame+1,2])
        ev_line.set_data(ev_traj[:frame+1,0], ev_traj[:frame+1,1])
        ev_line.set_3d_properties(ev_traj[:frame+1,2])
        pts = np.vstack((ch_traj[frame], ev_traj[frame]))
        min_pt, max_pt = pts.min(axis=0), pts.max(axis=0)
        center = (min_pt + max_pt) / 2
        half = np.max(max_pt - min_pt) / 2 + margin
        ax.set_xlim(center[0]-half, center[0]+half)
        ax.set_ylim(center[1]-half, center[1]+half)
        ax.set_zlim(center[2]-half, center[2]+half)
        set_axes_equal(ax)
        return ch_line, ev_line

    ani = FuncAnimation(fig, update, frames=len(ch_traj), interval=100, blit=False)
    path = os.path.join(save_dir, filename)
    ani.save(path, writer=PillowWriter(fps=10))
    plt.close()
    print(f"Saved animation GIF: {path}")

chaser_path, evader_path = load_latest_policies()

env = FightEnv(K_history=1)
obs_dim = env.observation_spaces['Agent 0'].shape[0]
act_dim = env.action_spaces['Agent 0'].shape[0]

chaser_net = CustomActorCritic(obs_dim, act_dim)
chaser_net.load_state_dict(torch.load(chaser_path, map_location='cpu'))
for p in chaser_net.parameters(): p.requires_grad = False

evader_net = CustomActorCritic(obs_dim, act_dim)
evader_net.load_state_dict(torch.load(evader_path, map_location='cpu'))
for p in evader_net.parameters(): p.requires_grad = False

ts = datetime.datetime.now().strftime('%Y%m%d-%H%M')
save_dir = os.path.join('plots', f'eval_fight_{ts}')
os.makedirs(save_dir, exist_ok=True)

print("Evaluating Fight agents for 100 episodes...")
trajectories, followed = evaluate_policies(chaser_net, evader_net, env, num_episodes=100)

follows_ch = [f[0] for f in followed]
best_idx  = int(np.argmax(follows_ch))
worst_idx = int(np.argmin(follows_ch))
print(f"Best episode #{best_idx+1} with followed={follows_ch[best_idx]}")
print(f"Worst episode #{worst_idx+1} with followed={follows_ch[worst_idx]}")

# Save plot and GIF
plot_trajectory(*trajectories[best_idx],  f"best_episode_{best_idx+1}",  save_dir)
plot_trajectory(*trajectories[worst_idx], f"worst_episode_{worst_idx+1}", save_dir)

animate_trajectory(*trajectories[best_idx],  f"best_episode_{best_idx+1}.gif",  save_dir)
animate_trajectory(*trajectories[worst_idx], f"worst_episode_{worst_idx+1}.gif", save_dir)

print(f"All outputs saved to {save_dir}")