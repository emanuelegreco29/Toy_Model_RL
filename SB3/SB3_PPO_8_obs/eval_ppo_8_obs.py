import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from stable_baselines3 import PPO, SAC, TD3

from env_sb3_ppo_8_obs import PointMassEnv


def load_latest_model(model_dir="models", algo="PPO"):
    algo_l = algo.lower()

    run_pattern = os.path.join(model_dir, f"{algo_l}_sb3_8_obs_*")
    run_dirs = [p for p in glob.glob(run_pattern) if os.path.isdir(p)]
    if run_dirs:
        latest_run_dir = max(run_dirs, key=os.path.getmtime)
        model_path = os.path.join(latest_run_dir, f"{algo_l}_model.zip")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found in latest run dir: {model_path}")
        print(f"Loading model: {model_path}")
        if algo.upper() == "PPO":
            return PPO.load(model_path)
        if algo.upper() == "SAC":
            return SAC.load(model_path)
        if algo.upper() == "TD3":
            return TD3.load(model_path)
        raise ValueError(f"Unknown algo: {algo}")

    flat_pattern = os.path.join(model_dir, f"{algo_l}_sb3_8_obs_*.zip")
    files = glob.glob(flat_pattern)
    if not files:
        raise FileNotFoundError(f"No model files matching {flat_pattern} and no run dirs matching {run_pattern}")
    latest = max(files, key=os.path.getmtime)
    print(f"Loading model: {latest}")
    if algo.upper() == "PPO":
        return PPO.load(latest)
    if algo.upper() == "SAC":
        return SAC.load(latest)
    if algo.upper() == "TD3":
        return TD3.load(latest)
    raise ValueError(f"Unknown algo: {algo}")


def evaluate_model(model, target, num_episodes=10, deterministic=True):
    trajectories = []
    targets = []
    rewards = []
    distances = []
    steps = []
    successes = []

    env = PointMassEnv()

    for _ in range(num_episodes):
        obs, _ = env.reset()

        # Fixed target per-episode (if provided). If target is None, env uses its own target sampling at reset().
        if target is not None:
            env.target = np.array(target, dtype=np.float32)

        ep_target = env.target.copy()

        traj = [env.state[:3].copy()]
        ep_reward = 0.0
        ep_steps = 0
        ep_success = 0

        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += float(reward)
            traj.append(env.state[:3].copy())
            ep_steps += 1

            if terminated or truncated:
                ep_success = 1 if terminated else 0
                final_dist = float(info.get("distance", np.linalg.norm(env.state[:3] - env.target)))
                break

        trajectories.append(np.array(traj, dtype=np.float32))
        targets.append(ep_target.astype(np.float32))
        rewards.append(float(ep_reward))
        distances.append(float(final_dist))
        steps.append(int(ep_steps))
        successes.append(int(ep_success))

    return trajectories, targets, rewards, distances, steps, successes


def plot_trajectory(traj, target):
    n = len(traj)
    colors = cm.viridis(np.linspace(0, 1, n))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for i in range(n - 1):
        seg = traj[i : i + 2]
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=colors[i])

    ax.scatter(*traj[0], color="green", s=50, label="Start")
    ax.scatter(*traj[-1], color="blue", s=50, label="End")

    tgt = np.asarray(target, dtype=float).reshape(3,)
    ax.scatter(tgt[0], tgt[1], tgt[2], color="red", marker="*", s=200, label="Target")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.title("Trajectory with color progression")
    plt.show()


# Configuration
ALGO = "TD3"  # "PPO"  # "SAC"  # "TD3"
NUM_EPISODES = 10
NEW_TARGET = None         # None => random target per episode (env.reset). Otherwise [x,y,z]
DETERMINISTIC = True      # set False if you want stochastic actions (mainly for SAC/TD3 exploration)
PLOT_RANDOM_TRAJ = True   # if True, plots one random episode trajectory

model = load_latest_model(model_dir="models", algo=ALGO)
trajectories, targets, rewards, distances, steps, successes = evaluate_model(
    model=model,
    target=NEW_TARGET,
    num_episodes=NUM_EPISODES,
    deterministic=DETERMINISTIC,
)

rewards = np.asarray(rewards, dtype=float)
distances = np.asarray(distances, dtype=float)
steps = np.asarray(steps, dtype=float)
successes = np.asarray(successes, dtype=float)

ddof = 1 if NUM_EPISODES >= 2 else 0

print("\n=== Evaluation metrics (mean +- std over episodes) ===")
print(f"Algo: {ALGO}")
print(f"Avg Reward: {np.mean(rewards):.2f} +- {np.std(rewards, ddof=ddof):.2f}")
print(f"Avg Steps/episode (↓): {np.mean(steps):.2f} +- {np.std(steps, ddof=ddof):.2f}")
print(f"Avg distance (↓): {np.mean(distances):.3f} +- {np.std(distances, ddof=ddof):.3f}")
print(f"Success ratio [%] (↑): {100.0 * np.mean(successes):.1f} +- {100.0 * np.std(successes, ddof=ddof):.1f}")

if PLOT_RANDOM_TRAJ:
    idx = random.randrange(NUM_EPISODES)
    print(f"\nPlotting trajectory for episode {idx + 1}")
    plot_trajectory(trajectories[idx], targets[idx])