import os
import sys
import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environments.tag_env import TagEnv
from algorithms.PPO.custom_ppo import CustomActorCritic


def load_latest_policies(base_dir="policies/Tag_SP"):
    folders = glob.glob(os.path.join(base_dir, "Tag_SelfPlay_*"))
    if not folders:
        raise FileNotFoundError(f"No Tag_SelfPlay folders found in {base_dir}")
    latest_folder = max(folders, key=os.path.getmtime)
    print(f"Loading policies from: {latest_folder}")
    files = glob.glob(os.path.join(latest_folder, "*_policy_ep*.pth"))
    chaser_file = max([f for f in files if "Agent 0" in f], key=os.path.getmtime)
    evader_file = max([f for f in files if "Agent 1" in f], key=os.path.getmtime)
    return chaser_file, evader_file


def evaluate_policies(chaser_net, evader_net, env, num_episodes=100):
    trajs = []
    followed = []
    returns = []

    for _ in range(num_episodes):
        obs, infos = env.reset()
        done = False

        ch_traj = []
        ev_traj = []

        ep_ret = {"Agent 0": 0.0, "Agent 1": 0.0}

        while not done:
            ch_traj.append(env.states["Agent 0"][:3].copy())
            ev_traj.append(env.states["Agent 1"][:3].copy())

            actions = {}
            for name, net in [("Agent 0", chaser_net), ("Agent 1", evader_net)]:
                o = torch.tensor(obs[name], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    mean, log_std, _ = net(o)
                actions[name] = mean.squeeze().numpy()

            obs, rews, dones, infos = env.step(actions)

            for ag in ("Agent 0", "Agent 1"):
                ep_ret[ag] += float(rews[ag])

            done = any(dones.values())

        trajs.append((np.array(ch_traj), np.array(ev_traj)))

        fch = int(infos["Agent 0"]["followed"])
        fev = int(infos["Agent 1"]["followed"])
        followed.append((fch, fev))

        returns.append((float(ep_ret["Agent 0"]), float(ep_ret["Agent 1"])))

    return trajs, followed, returns


def _add_start_end_markers(ax, traj, label_prefix, start_marker="o", end_marker="X"):
    if traj.shape[0] < 1:
        return

    start = traj[0]
    end = traj[-1]

    ax.scatter(
        [start[0]],
        [start[1]],
        [start[2]],
        marker=start_marker,
        s=60,
        label=f"{label_prefix} start",
    )
    ax.scatter(
        [end[0]],
        [end[1]],
        [end[2]],
        marker=end_marker,
        s=70,
        label=f"{label_prefix} end",
    )


def plot_trajectory(ch_traj, ev_traj, title, save_dir):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(ch_traj[:, 0], ch_traj[:, 1], ch_traj[:, 2], label="Chaser path")
    ax.plot(ev_traj[:, 0], ev_traj[:, 1], ev_traj[:, 2], linestyle="--", label="Evader path")

    _add_start_end_markers(ax, ch_traj, "Chaser", start_marker="o", end_marker="X")
    _add_start_end_markers(ax, ev_traj, "Evader", start_marker="^", end_marker="s")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, f"{title}.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved trajectory plot: {path}")


def set_axes_equal(ax):
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
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=30, azim=45)

    ch_line, = ax.plot([], [], [], lw=2, label="Chaser path")
    ev_line, = ax.plot([], [], [], lw=2, linestyle="--", label="Evader path")

    ch_start = ch_traj[0] if ch_traj.shape[0] > 0 else np.array([0.0, 0.0, 0.0])
    ev_start = ev_traj[0] if ev_traj.shape[0] > 0 else np.array([0.0, 0.0, 0.0])
    ch_end = ch_traj[-1] if ch_traj.shape[0] > 0 else np.array([0.0, 0.0, 0.0])
    ev_end = ev_traj[-1] if ev_traj.shape[0] > 0 else np.array([0.0, 0.0, 0.0])

    ax.scatter([ch_start[0]], [ch_start[1]], [ch_start[2]], marker="o", s=60, label="Chaser start")
    ax.scatter([ev_start[0]], [ev_start[1]], [ev_start[2]], marker="^", s=60, label="Evader start")

    ch_end_sc = ax.scatter([ch_end[0]], [ch_end[1]], [ch_end[2]], marker="X", s=70, label="Chaser end")
    ev_end_sc = ax.scatter([ev_end[0]], [ev_end[1]], [ev_end[2]], marker="s", s=70, label="Evader end")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    margin = 0.5

    def update(frame):
        ch_line.set_data(ch_traj[: frame + 1, 0], ch_traj[: frame + 1, 1])
        ch_line.set_3d_properties(ch_traj[: frame + 1, 2])

        ev_line.set_data(ev_traj[: frame + 1, 0], ev_traj[: frame + 1, 1])
        ev_line.set_3d_properties(ev_traj[: frame + 1, 2])

        pts = np.vstack((ch_traj[frame], ev_traj[frame]))
        min_pt = pts.min(axis=0)
        max_pt = pts.max(axis=0)
        center = (min_pt + max_pt) / 2.0
        half_range = np.max(max_pt - min_pt) / 2.0 + margin

        ax.set_xlim(center[0] - half_range, center[0] + half_range)
        ax.set_ylim(center[1] - half_range, center[1] + half_range)
        ax.set_zlim(center[2] - half_range, center[2] + half_range)
        set_axes_equal(ax)

        return ch_line, ev_line, ch_end_sc, ev_end_sc

    ani = FuncAnimation(fig, update, frames=len(ch_traj), interval=100, blit=False)
    path = os.path.join(save_dir, filename)
    ani.save(path, writer=PillowWriter(fps=10))
    plt.close()
    print(f"Saved animation GIF: {path}")


chaser_path, evader_path = load_latest_policies()

env = TagEnv(K_history=1)
obs_dim = env.observation_spaces["Agent 0"].shape[0]
act_dim = env.action_spaces["Agent 0"].shape[0]

chaser_net = CustomActorCritic(obs_dim, act_dim)
chaser_net.load_state_dict(torch.load(chaser_path, map_location="cpu"))
for p in chaser_net.parameters():
    p.requires_grad = False

evader_net = CustomActorCritic(obs_dim, act_dim)
evader_net.load_state_dict(torch.load(evader_path, map_location="cpu"))
for p in evader_net.parameters():
    p.requires_grad = False

ts = datetime.datetime.now().strftime("%Y%m%d-%H%M")
save_dir = os.path.join("plots/Tag", f"eval_tag_{ts}")
os.makedirs(save_dir, exist_ok=True)

num_episodes = 100
print(f"Evaluating Tag agents for {num_episodes} episodes...")
trajectories, followed, returns = evaluate_policies(
    chaser_net, evader_net, env, num_episodes=num_episodes
)

follows_ch = np.array([f[0] for f in followed], dtype=np.float32)
follows_ev = np.array([f[1] for f in followed], dtype=np.float32)

rets_ch = np.array([r[0] for r in returns], dtype=np.float32)
rets_ev = np.array([r[1] for r in returns], dtype=np.float32)
rets_mean = 0.5 * (rets_ch + rets_ev)

best_idx = int(np.argmax(follows_ch))
worst_idx = int(np.argmin(follows_ch))
print(f"Best episode #{best_idx + 1} with followed={int(follows_ch[best_idx])}")
print(f"Worst episode #{worst_idx + 1} with followed={int(follows_ch[worst_idx])}")

plot_trajectory(*trajectories[best_idx], f"best_episode_{best_idx + 1}", save_dir)
plot_trajectory(*trajectories[worst_idx], f"worst_episode_{worst_idx + 1}", save_dir)

animate_trajectory(*trajectories[best_idx], f"best_episode_{best_idx + 1}.gif", save_dir)
animate_trajectory(*trajectories[worst_idx], f"worst_episode_{worst_idx + 1}.gif", save_dir)

print("\n===== Evaluation summary (mean ± std over episodes) =====")
print(f"Avg reward (episode return) Agent 0: {float(np.mean(rets_ch)):.6f} ± {float(np.std(rets_ch, ddof=0)):.6f}")
print(f"Avg reward (episode return) Agent 1: {float(np.mean(rets_ev)):.6f} ± {float(np.std(rets_ev, ddof=0)):.6f}")
print(f"Avg reward (mean of agents):         {float(np.mean(rets_mean)):.6f} ± {float(np.std(rets_mean, ddof=0)):.6f}")
print(f"Avg totally behind Agent 0:          {float(np.mean(follows_ch)):.6f} ± {float(np.std(follows_ch, ddof=0)):.6f}")
print(f"Avg totally behind Agent 1:          {float(np.mean(follows_ev)):.6f} ± {float(np.std(follows_ev, ddof=0)):.6f}")

print(f"\nAll outputs saved to {save_dir}")