import os
import sys
import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
from matplotlib.animation import FuncAnimation, PillowWriter

# ensure relative imports work when running from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environments.tag_shoot_env import TagShootEnv
from algorithms.IPPO.models import EnhancedActorCritic

# Helper functions
def find_latest_ckpt(base_dir="policies"):
    fixed_dirs = glob.glob(os.path.join(base_dir, "Tag_Shoot_IPPO", "*"))
    if fixed_dirs:
        latest = max(fixed_dirs, key=os.path.getmtime)
        print(f"Found training run: {latest}")
    else:
        # Fallback to original directory structure
        runs = glob.glob(os.path.join(base_dir, "Tag_Shoot_IPPO", "*"))
        if not runs:
            raise FileNotFoundError(f"No runs found in {base_dir}")
        latest = max(runs, key=os.path.getmtime)
    
    # Try per-agent checkpoints
    ckpts = glob.glob(os.path.join(latest, "*_policy_*.pth"))
    ch, ev = None, None
    
    if ckpts:
        chs = [p for p in ckpts if "Agent 0" in os.path.basename(p)]
        evs = [p for p in ckpts if "Agent 1" in os.path.basename(p)]
        if chs and evs:
            # Get the latest checkpoint for each agent
            ch = max(chs, key=lambda x: int(x.split('upd')[1].split('.pth')[0]) if 'upd' in x else 0)
            ev = max(evs, key=lambda x: int(x.split('upd')[1].split('.pth')[0]) if 'upd' in x else 0)
            print(f"Using Agent 0 checkpoint: {os.path.basename(ch)}")
            print(f"Using Agent 1 checkpoint: {os.path.basename(ev)}")
    
    # Fallback: single shared policy
    if ch is None or ev is None:
        shared = glob.glob(os.path.join(latest, "*shared*_*.pth")) or glob.glob(os.path.join(latest, "*policy_*.pth"))
        if not shared:
            raise FileNotFoundError(f"No checkpoints found in {latest}")
        shared = max(shared, key=os.path.getmtime)
        ch = shared
        ev = shared
        print(f"Using shared checkpoint: {os.path.basename(shared)}")
    
    return ch, ev, latest

def policy_from_ckpt(path, obs_dim, act_dim, device="cpu"):
    """Load policy from checkpoint"""
    net = EnhancedActorCritic(obs_dim, act_dim)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    net.load_state_dict(state_dict)
    print(f"Loaded model from {os.path.basename(path)}")
    
    net.eval()
    return net.to(device)

# Plotting helper functions
def set_axes_cube(ax, center, radius):
    """Set 3D axes limits to create a cube around the center point."""
    cx, cy, cz = center
    r = float(radius)
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_zlim(cz - r, cz + r)
    
def plot_traj_with_hits(ch, ev, hits_on_0, hits_on_1, title, outdir, episode_info=None):
    """
    hits_on_0 / hits_on_1: list of indices (frame numbers) where that agent got hit
    """
    fig = plt.figure(figsize=(8, 7), dpi=130)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(ch[:, 0], ch[:, 1], ch[:, 2], label="Agent 0", linewidth=2)
    ax.plot(ev[:, 0], ev[:, 1], ev[:, 2], "--", label="Agent 1", linewidth=2)

    ax.scatter(ch[0, 0], ch[0, 1], ch[0, 2], s=80, marker="o", label="A0 start", alpha=0.8)
    ax.scatter(ev[0, 0], ev[0, 1], ev[0, 2], s=80, marker="o", label="A1 start", alpha=0.8)
    ax.scatter(ch[-1, 0], ch[-1, 1], ch[-1, 2], s=90, marker="^", label="A0 end", alpha=0.8)
    ax.scatter(ev[-1, 0], ev[-1, 1], ev[-1, 2], s=90, marker="^", label="A1 end", alpha=0.8)

    # X markers where hits occurred
    def clamp_idx(i, N): return max(0, min(int(i), N - 1))
    if hits_on_0:
        idxs = [clamp_idx(i, len(ch)) for i in hits_on_0]
        ax.scatter(ch[idxs, 0], ch[idxs, 1], ch[idxs, 2], marker="x", s=120, linewidths=3, 
                  color='red', label="Hit on A0", alpha=0.9)
    if hits_on_1:
        idxs = [clamp_idx(i, len(ev)) for i in hits_on_1]
        ax.scatter(ev[idxs, 0], ev[idxs, 1], ev[idxs, 2], marker="x", s=120, linewidths=3, 
                  color='red', label="Hit on A1", alpha=0.9)

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    
    # Enhanced title with episode info
    if episode_info:
        title_text = f"{title}\n{episode_info}"
    else:
        title_text = title
    ax.set_title(title_text, fontsize=12, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    # fit cube tight on both trajs
    data = np.vstack([ch, ev])
    mins, maxs = data.min(0), data.max(0)
    center = 0.5 * (mins + maxs)
    radius = max((maxs - mins) / 2.0)
    radius = float(max(radius, 2.0))  # Minimum radius for better visibility
    set_axes_cube(ax, center, radius)

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{title}.png")
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', dpi=130)
    plt.close()
    print(f"Saved {path}")

def animate_traj_follow_zoom(ch, ev, hits_on_0, hits_on_1, title, outdir,
                             window=60, margin=2.0, min_radius=4.0, fps=20, dpi=120):
    """
    Enhanced GIF animation with better following and hit visualization
    """
    T = min(len(ch), len(ev))
    ch = np.asarray(ch[:T], dtype=np.float64)
    ev = np.asarray(ev[:T], dtype=np.float64)

    hits_on_0 = [i for i in hits_on_0 if 0 <= i < T]
    hits_on_1 = [i for i in hits_on_1 if 0 <= i < T]

    fig = plt.figure(figsize=(7, 7), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title, fontsize=12)

    # lines and points
    a0_line, = ax.plot([], [], [], lw=2.5, label="Agent 0", alpha=0.8)
    a1_line, = ax.plot([], [], [], lw=2.5, linestyle="--", label="Agent 1", alpha=0.8)
    a0_pt, = ax.plot([], [], [], marker="o", markersize=8, linestyle="None", color='blue')
    a1_pt, = ax.plot([], [], [], marker="o", markersize=8, linestyle="None", color='orange')

    # Hit markers
    hit_a0_scat = ax.scatter([], [], [], marker="x", s=120, linewidths=3, color='red', alpha=0.9)
    hit_a1_scat = ax.scatter([], [], [], marker="x", s=120, linewidths=3, color='red', alpha=0.9)

    ax.legend(loc="upper right")

    # smooth camera tracking
    center_ema = None
    radius_ema = None
    alpha = 0.25  # Camera smoothing

    def update(i):
        nonlocal center_ema, radius_ema
        i = int(i)
        start = max(0, i - window + 1)

        # Update trajectory lines
        a0_hist = ch[:i + 1]
        a1_hist = ev[:i + 1]
        a0_line.set_data(a0_hist[:, 0], a0_hist[:, 1])
        a0_line.set_3d_properties(a0_hist[:, 2])
        a1_line.set_data(a1_hist[:, 0], a1_hist[:, 1])
        a1_line.set_3d_properties(a1_hist[:, 2])

        # Update current position markers
        a0_pt.set_data([ch[i, 0]], [ch[i, 1]]); a0_pt.set_3d_properties([ch[i, 2]])
        a1_pt.set_data([ev[i, 0]], [ev[i, 1]]); a1_pt.set_3d_properties([ev[i, 2]])

        # Update hit markers (only hits that have occurred so far)
        a0_idxs = [j for j in hits_on_0 if j <= i]
        a1_idxs = [j for j in hits_on_1 if j <= i]
        
        if a0_idxs:
            a0p = ch[a0_idxs]
            hit_a0_scat._offsets3d = (a0p[:, 0], a0p[:, 1], a0p[:, 2])
        else:
            hit_a0_scat._offsets3d = ([], [], [])
            
        if a1_idxs:
            a1p = ev[a1_idxs]
            hit_a1_scat._offsets3d = (a1p[:, 0], a1p[:, 1], a1p[:, 2])
        else:
            hit_a1_scat._offsets3d = ([], [], [])

        # Dynamic camera: focus on the action
        current_center = 0.5 * (ch[i] + ev[i])
        
        # Consider recent trajectory for bounding
        w0 = ch[start:i + 1]
        w1 = ev[start:i + 1]
        data = np.vstack([w0, w1])
        mins, maxs = data.min(0), data.max(0)
        half_ranges = 0.5 * (maxs - mins)
        r_local = float(np.max(half_ranges) + margin)
        
        # Also consider current separation
        r_pair = float(np.linalg.norm(ch[i] - ev[i]) * 0.4 + margin)
        r = max(r_local, r_pair, min_radius)

        # Smooth camera movement
        if center_ema is None:
            center_ema = current_center
            radius_ema = r
        else:
            center_ema = (1 - alpha) * center_ema + alpha * current_center
            radius_ema = (1 - alpha) * radius_ema + alpha * r

        set_axes_cube(ax, center_ema, radius_ema)
        return a0_line, a1_line, a0_pt, a1_pt

    ani = FuncAnimation(fig, update, frames=T, interval=int(1000 / fps), blit=False)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{title}.gif")
    try:
        ani.save(path, writer=PillowWriter(fps=fps))
        print(f"Saved {path}")
    except Exception as e:
        print(f"Failed to save animation: {e}")
    finally:
        plt.close()

# Evaluation run
ch_path, ev_path, run_dir = find_latest_ckpt()
env = TagShootEnv()

obs_dim = env.observation_spaces['Agent 0'].shape[0]
act_dim = env.action_spaces['Agent 0'].shape[0]
dev = "cpu"

print(f"Environment obs_dim: {obs_dim}, act_dim: {act_dim}")

# Load policies with fallback
ch_net = policy_from_ckpt(ch_path, obs_dim, act_dim, device=dev)
ev_net = policy_from_ckpt(ev_path, obs_dim, act_dim, device=dev)

ts = datetime.datetime.now().strftime('%Y%m%d-%H%M')
outdir = os.path.join("plots", "Tag_Shoot_IPPO", f"eval_{ts}")
os.makedirs(outdir, exist_ok=True)

num_eps = 50
best_metric = -np.inf
best_idx = 0
episodes = []  # (A0_traj, A1_traj, hits_on_0, hits_on_1, metric, episode_stats)

print(f"Running {num_eps} evaluation episodes...")

for ep in range(num_eps):
    obs, infos = env.reset()
    done = False
    a0_traj, a1_traj = [], []
    hp0_hist, hp1_hist = [], []

    # Initial HP snapshot
    hp0_hist.append(int(env.hp['Agent 0']))
    hp1_hist.append(int(env.hp['Agent 1']))

    step_count = 0
    while not done and step_count < 2000:  # Safety limit
        a0_traj.append(env.states['Agent 0'][:3].copy())
        a1_traj.append(env.states['Agent 1'][:3].copy())

        actions = {}
        with torch.no_grad():
            for name, net in [('Agent 0', ch_net), ('Agent 1', ev_net)]:
                o = torch.tensor(obs[name], dtype=torch.float32).unsqueeze(0)
                a, _, _, _ = net.get_action_and_value(o)
                low = torch.as_tensor(env.action_spaces[name].low, dtype=torch.float32)
                high = torch.as_tensor(env.action_spaces[name].high, dtype=torch.float32)
                a = torch.clamp(a.squeeze(0), low, high)
                actions[name] = a.cpu().numpy()

        obs, rews, dones, infos = env.step(actions)
        hp0_hist.append(int(env.hp['Agent 0']))
        hp1_hist.append(int(env.hp['Agent 1']))

        done = any(dones.values())
        step_count += 1

    a0_traj = np.array(a0_traj, dtype=np.float64)
    a1_traj = np.array(a1_traj, dtype=np.float64)
    hp0_hist = np.array(hp0_hist, dtype=np.int32)
    hp1_hist = np.array(hp1_hist, dtype=np.int32)

    # Detect hits: indices where HP drops
    hits_on_0 = np.where(hp0_hist[1:] < hp0_hist[:-1])[0].tolist()
    hits_on_1 = np.where(hp1_hist[1:] < hp1_hist[:-1])[0].tolist()
    
    # Enhanced metrics
    total_locks = sum(infos[ag].get('Total Lock', 0) for ag in env.agents)
    total_hits = len(hits_on_0) + len(hits_on_1)
    final_hp_diff = int(env.hp['Agent 0']) - int(env.hp['Agent 1'])
    engagement_metric = total_locks + total_hits * 10 + abs(final_hp_diff) * 0.1

    episode_stats = {
        'total_locks': total_locks,
        'total_hits': total_hits,
        'hits_on_0': len(hits_on_0),
        'hits_on_1': len(hits_on_1),
        'final_hp': (int(env.hp['Agent 0']), int(env.hp['Agent 1'])),
        'episode_length': len(a0_traj),
        'engagement_metric': engagement_metric
    }

    episodes.append((a0_traj, a1_traj, hits_on_0, hits_on_1, engagement_metric, episode_stats))

    if engagement_metric > best_metric:
        best_metric = engagement_metric
        best_idx = ep

    if (ep + 1) % 10 == 0:
        print(f"Completed {ep + 1}/{num_eps} episodes")

print(f"\nEvaluation completed!")
print(f"Best episode #{best_idx + 1} with engagement metric={best_metric:.2f}")

# Plot and animate best episode
best = episodes[best_idx]
best_info = f"Locks: {best[5]['total_locks']} | Hits: {best[5]['total_hits']} | HP: {best[5]['final_hp']}"
plot_traj_with_hits(best[0], best[1], best[2], best[3], 
                    f"best_episode_{best_idx + 1}", outdir, best_info)
animate_traj_follow_zoom(best[0], best[1], best[2], best[3], 
                        f"best_episode_{best_idx + 1}_follow", outdir)

# Plot and animate worst episode
worst_idx = int(np.argmin([e[4] for e in episodes]))
worst = episodes[worst_idx]
worst_info = f"Locks: {worst[5]['total_locks']} | Hits: {worst[5]['total_hits']} | HP: {worst[5]['final_hp']}"
plot_traj_with_hits(worst[0], worst[1], worst[2], worst[3], 
                    f"worst_episode_{worst_idx + 1}", outdir, worst_info)
animate_traj_follow_zoom(worst[0], worst[1], worst[2], worst[3], 
                        f"worst_episode_{worst_idx + 1}_follow", outdir)

# Print summary statistics
print(f"\n=== EVALUATION SUMMARY ===")
print(f"Total episodes: {num_eps}")
total_hits = sum(len(e[2]) + len(e[3]) for e in episodes)
total_locks = sum(e[5]['total_locks'] for e in episodes)
avg_length = np.mean([e[5]['episode_length'] for e in episodes])
    
print(f"Total hits across all episodes: {total_hits}")
print(f"Total locks across all episodes: {total_locks}")
print(f"Average episode length: {avg_length:.1f} steps")
print(f"Outputs saved to {outdir}")
print(f"Models loaded from {run_dir}")