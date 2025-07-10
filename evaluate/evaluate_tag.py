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

from environments.tag_env import TagEnv
from algorithms.PPO.custom_ppo import CustomActorCritic

def load_latest_policies(base_dir='policies/Tag_SP'):
    folders = glob.glob(os.path.join(base_dir, 'Tag_SelfPlay_*'))
    if not folders:
        raise FileNotFoundError(f"No Tag_SelfPlay folders found in {base_dir}")
    latest_folder = max(folders, key=os.path.getmtime)
    print(f"Loading policies from: {latest_folder}")
    files = glob.glob(os.path.join(latest_folder, '*_policy_ep*.pth'))
    chaser_file = max([f for f in files if 'Agent 0' in f], key=os.path.getmtime)
    evader_file = max([f for f in files if 'Agent 1' in f], key=os.path.getmtime)
    return chaser_file, evader_file

def evaluate_policies(chaser_net, evader_net, env, num_episodes=100):
    trajs = []
    followed = []
    for ep in range(num_episodes):
        obs, infos = env.reset()
        done = False
        ch_traj = []
        ev_traj = []
        while not done:
            ch_traj.append(env.states['Agent 0'][:3].copy())
            ev_traj.append(env.states['Agent 1'][:3].copy())
            actions = {}
            for name, net in [('Agent 0', chaser_net), ('Agent 1', evader_net)]:
                o = torch.tensor(obs[name], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    mean, log_std, _ = net(o)
                actions[name] = mean.squeeze().numpy()
            obs, rews, dones, infos = env.step(actions)
            done = any(dones.values())
        trajs.append((np.array(ch_traj), np.array(ev_traj)))
        fch = infos['Agent 0']['followed']
        fev = infos['Agent 1']['followed']
        followed.append((fch, fev))
    return trajs, followed

def plot_trajectory(ch_traj, ev_traj, title, save_dir):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(ch_traj[:,0], ch_traj[:,1], ch_traj[:,2], label='Chaser')
    ax.plot(ev_traj[:,0], ev_traj[:,1], ev_traj[:,2], linestyle='--', label='Evader')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, f"{title}.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved trajectory plot: {path}")

def animate_trajectory(ch_traj, ev_traj, filename, save_dir):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ch_line, = ax.plot([], [], [], lw=2, label='Chaser')
    ev_line, = ax.plot([], [], [], lw=2, linestyle='--', label='Evader')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()

    def update(frame):
        # aggiorna dati fino al frame corrente
        ch_line.set_data(ch_traj[:frame+1,0], ch_traj[:frame+1,1])
        ch_line.set_3d_properties(ch_traj[:frame+1,2])
        ev_line.set_data(ev_traj[:frame+1,0], ev_traj[:frame+1,1])
        ev_line.set_3d_properties(ev_traj[:frame+1,2])
        # regola i limiti in modo dinamico
        all_pts = np.vstack([ch_traj[:frame+1], ev_traj[:frame+1]])
        mins = all_pts.min(axis=0) - 0.5
        maxs = all_pts.max(axis=0) + 0.5
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        return ch_line, ev_line

    ani = FuncAnimation(fig, update, frames=len(ch_traj), interval=100, blit=False)
    path = os.path.join(save_dir, filename)
    ani.save(path, writer=PillowWriter(fps=10))
    plt.close()
    print(f"Saved animation GIF: {path}")


chaser_path, evader_path = load_latest_policies()
env = TagEnv(K_history=1)
obs_dim = env.observation_spaces['Agent 0'].shape[0]
act_dim = env.action_spaces['Agent 0'].shape[0]

chaser_net = CustomActorCritic(obs_dim, act_dim)
chaser_net.load_state_dict(torch.load(chaser_path, map_location='cpu'))
for p in chaser_net.parameters(): p.requires_grad = False

evader_net = CustomActorCritic(obs_dim, act_dim)
evader_net.load_state_dict(torch.load(evader_path, map_location='cpu'))
for p in evader_net.parameters(): p.requires_grad = False

ts = datetime.datetime.now().strftime('%Y%m%d-%H%M')
save_dir = os.path.join('plots', f'eval_tag_{ts}')
os.makedirs(save_dir, exist_ok=True)

print(f"Evaluating Tag agents for 100 episodes...")
trajectories, followed = evaluate_policies(chaser_net, evader_net, env, num_episodes=100)

follows_ch = [f[0] for f in followed]
best_idx  = int(np.argmax(follows_ch))
worst_idx = int(np.argmin(follows_ch))
print(f"Best episode #{best_idx+1} with followed={follows_ch[best_idx]}")
print(f"Worst episode #{worst_idx+1} with followed={follows_ch[worst_idx]}")

plot_trajectory(*trajectories[best_idx],  f"best_episode_{best_idx+1}",  save_dir)
plot_trajectory(*trajectories[worst_idx], f"worst_episode_{worst_idx+1}", save_dir)

animate_trajectory(*trajectories[best_idx],  f"best_episode_{best_idx+1}.gif",  save_dir)
animate_trajectory(*trajectories[worst_idx], f"worst_episode_{worst_idx+1}.gif", save_dir)

print(f"All outputs saved to {save_dir}")