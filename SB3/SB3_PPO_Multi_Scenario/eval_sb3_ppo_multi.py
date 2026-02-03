import os, glob, random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from stable_baselines3 import PPO, SAC, TD3
import datetime
import torch

from env_sb3_ppo_multi import PointMassEnv


def load_latest_model(model_root='models', algo='PPO', device='cpu'):
    algo = algo.upper()
    algo_l = algo.lower()

    # 1) preferisci la struttura creata dal train: models/{algo}_sb3_multi_YYYY.../{algo}_model.zip
    run_dirs = glob.glob(os.path.join(model_root, f"{algo_l}_sb3_multi_*"))
    run_dirs = [d for d in run_dirs if os.path.isdir(d)]
    if run_dirs:
        latest_dir = max(run_dirs, key=os.path.getmtime)
        cand = os.path.join(latest_dir, f"{algo_l}_model.zip")
        if os.path.exists(cand):
            model_path = cand
        else:
            model_path = None
    else:
        model_path = None

    # 2) fallback: cerca ricorsivamente qualsiasi .../{algo}_model.zip
    if model_path is None:
        files = glob.glob(os.path.join(model_root, "**", f"{algo_l}_model.zip"), recursive=True)
        if not files:
            raise FileNotFoundError(f"No {algo_l}_model.zip found under {model_root}/")
        model_path = max(files, key=os.path.getmtime)

    print(f"Loading {algo} model: {model_path}")

    cls = {"PPO": PPO, "SAC": SAC, "TD3": TD3}[algo]
    return cls.load(model_path, device=device)


def evaluate_model(model, num_episodes=10):
    trajectories, target_trajs, rewards, distances, followed, action_logs = [], [], [], [], [], []
    env = PointMassEnv()
    for ep in range(num_episodes):
        obs, _ = env.reset()
        agent_path, target_path = [], []
        ep_reward = 0.0
        ep_action_log = []
        while True:
            agent_path.append(env.state[:3].copy())
            target_path.append(env.target_state[:3].copy())

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            # Log per-step info
            distance = info.get('distance', np.nan)
            behind_flag = env._is_behind()
            action_str = (
                f"delta_v: {action[0]:.3f}, "
                f"delta_yaw: {action[1]:.3f}, "
                f"delta_pitch: {action[2]:.3f}"
            )
            ep_action_log.append({
                'distance': distance,
                'behind': behind_flag,
                'reward': reward,
                'action_str': action_str
            })

            ep_reward += reward
            if terminated or truncated:
                agent_path.append(env.state[:3].copy())
                target_path.append(env.target_state[:3].copy())
                rewards.append(ep_reward)
                distances.append(info.get('distance', np.nan))
                followed.append(info.get('totally_behind', info.get('followed', np.nan)))
                break

        trajectories.append(np.array(agent_path))
        target_trajs.append(np.array(target_path))
        action_logs.append(ep_action_log)
    return trajectories, target_trajs, rewards, distances, followed, action_logs


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
    
def plot_full_trajectory_3d(
    agent_traj,
    target_traj,
    filename="trajectory_overall.png",
    title="Trajectory (overall view)",
    save_dir="plots",
    pad=1.0,
    elev=20,
    azim=-60,
):
    os.makedirs(save_dir, exist_ok=True)

    agent_traj = np.asarray(agent_traj, dtype=float)
    target_traj = np.asarray(target_traj, dtype=float)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Trajectories
    ax.plot(agent_traj[:, 0], agent_traj[:, 1], agent_traj[:, 2], lw=2, color="blue", label="Agent trajectory")
    ax.plot(target_traj[:, 0], target_traj[:, 1], target_traj[:, 2], lw=2, color="orange", label="Target trajectory")

    # Start / End markers
    ax.scatter(agent_traj[0, 0], agent_traj[0, 1], agent_traj[0, 2], s=50, color="blue", marker="o", label="Agent start")
    ax.scatter(agent_traj[-1, 0], agent_traj[-1, 1], agent_traj[-1, 2], s=80, color="blue", marker="X", label="Agent end")

    ax.scatter(target_traj[0, 0], target_traj[0, 1], target_traj[0, 2], s=50, color="orange", marker="o", label="Target start")
    ax.scatter(target_traj[-1, 0], target_traj[-1, 1], target_traj[-1, 2], s=80, color="orange", marker="X", label="Target end")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    # Overall centered view (bounds from whole episode, equalized box)
    pts = np.vstack([agent_traj, target_traj])
    mins = pts.min(axis=0) - pad
    maxs = pts.max(axis=0) + pad
    center = (mins + maxs) / 2.0
    span = (maxs - mins)
    max_range = float(np.max(span)) if span.size else 1.0

    ax.set_xlim(center[0] - max_range / 2.0, center[0] + max_range / 2.0)
    ax.set_ylim(center[1] - max_range / 2.0, center[1] + max_range / 2.0)
    ax.set_zlim(center[2] - max_range / 2.0, center[2] + max_range / 2.0)

    # Camera
    ax.view_init(elev=elev, azim=azim)

    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=200)
    plt.close(fig)


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

ALGO = "PPO"  # "PPO" | "SAC" | "TD3"
DEVICE = "auto"

model = load_latest_model(algo=ALGO, device=DEVICE)
print("Model loaded successfully.")
if ALGO == "PPO":
    log_std = model.policy.log_std.detach().cpu().numpy()
    std = torch.exp(model.policy.log_std).detach().cpu().numpy()
    print("Log-std per dimensione d’azione:", log_std)
    print("Std per dimensione d’azione:   ", std)
timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M')
save_dir = os.path.join('plots', f"sb3_multi_eval_{timestamp}")
os.makedirs(save_dir, exist_ok=True)

num_ep = 100
print(f"Evaluating model for {num_ep} episodes...")
trajectories, target_trajs, rewards, distances, followed, action_logs = evaluate_model(model, num_ep)

rewards_arr = np.asarray(rewards, dtype=float)
dist_arr = np.asarray(distances, dtype=float)
behind_arr = np.asarray(followed, dtype=float)  # qui è totally_behind

def mean_std(x: np.ndarray):
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    if x.size == 1:
        return float(np.mean(x)), 0.0
    return float(np.mean(x)), float(np.std(x, ddof=1))

m_rew, s_rew = mean_std(rewards_arr)
m_dist, s_dist = mean_std(dist_arr)
m_b, s_b = mean_std(behind_arr)

print(
    f"Avg Reward: {m_rew:.2f} ± {s_rew:.2f} | "
    f"Final Distance: {m_dist:.3f} ± {s_dist:.3f} | "
    f"Totally Behind: {m_b:.2f} ± {s_b:.2f}"
)

best_idx = int(np.argmax(rewards))
worst_idx = int(np.argmin(rewards))
print(f"Best Episode {best_idx+1} | Reward: {rewards[best_idx]:.2f} | Followed target for: {followed[best_idx]}")
print(f"Worst Episode {worst_idx+1} | Reward: {rewards[worst_idx]:.2f} | Followed target for: {followed[worst_idx]}")

print("Saving action logs for best and worst episodes...")
for idx, filename in [(best_idx, 'best_actions.txt'), (worst_idx, 'worst_actions.txt')]:
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'w') as f:
        for i, log in enumerate(action_logs[idx]):
            behind = 'yes' if log['behind'] else 'no'
            f.write(
                f"Step: {i} | "
                f"Distance: {log['distance']:.3f} | "
                f"Behind: {behind} | "
                f"Reward: {log['reward']:.3f} | "
                f"Action: {log['action_str']}\n"
            )

print("Plotting coordinate time-series...")
plot_coordinates(trajectories[best_idx], target_trajs[best_idx], title_prefix='Best_Episode', save_dir=save_dir)
plot_coordinates(trajectories[worst_idx], target_trajs[worst_idx], title_prefix='Worst_Episode', save_dir=save_dir)

print("Plotting speed time-series...")
plot_velocities(trajectories[best_idx], target_trajs[best_idx], title_prefix='Best_Episode', save_dir=save_dir)
plot_velocities(trajectories[worst_idx], target_trajs[worst_idx], title_prefix='Worst_Episode', save_dir=save_dir)

print("Plotting 3D overall trajectories (PNG)...")
plot_full_trajectory_3d(
    trajectories[best_idx],
    target_trajs[best_idx],
    filename="best_episode_overall.png",
    title="Best Episode - Overall 3D Trajectory",
    save_dir=save_dir,
    pad=1.0,
)
plot_full_trajectory_3d(
    trajectories[worst_idx],
    target_trajs[worst_idx],
    filename="worst_episode_overall.png",
    title="Worst Episode - Overall 3D Trajectory",
    save_dir=save_dir,
    pad=1.0,
)

print("Creating animations...")
animate_trajectory(trajectories[best_idx], target_trajs[best_idx], filename='best_episode.gif', save_dir=save_dir)
animate_trajectory(trajectories[worst_idx], target_trajs[worst_idx], filename='worst_episode.gif', save_dir=save_dir)
print(f"Plots and animations saved in {save_dir}")
