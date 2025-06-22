import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from stable_baselines3 import PPO
from env_sb3_ppo_8_obs import PointMassEnv

def load_latest_model(model_dir='models', prefix='ppo_sb3_8_obs_', ext='.zip'):
    pattern = os.path.join(model_dir, f"{prefix}*{ext}")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No model files matching {pattern}")
    latest = max(files, key=os.path.getmtime)
    print(f"Loading model: {latest}")
    return PPO.load(latest)


def evaluate_model(model, target, num_episodes=10):
    trajectories = []
    rewards = []
    distances = []

    env = PointMassEnv()
    env.target = np.array(target, dtype=np.float32)

    for ep in range(num_episodes):
        obs, _ = env.reset()
        traj = [env.state[:3].copy()]
        ep_reward = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            traj.append(env.state[:3].copy())
            if terminated or truncated:
                break
        trajectories.append(np.array(traj))
        rewards.append(ep_reward)
        distances.append(info.get('distance', np.linalg.norm(env.state[:3] - env.target)))

    return trajectories, rewards, distances


def plot_trajectory(traj, target):
    n = len(traj)
    colors = cm.viridis(np.linspace(0,1,n))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(n-1):
        seg = traj[i:i+2]
        ax.plot(seg[:,0], seg[:,1], seg[:,2], color=colors[i])
    ax.scatter(*traj[0], color='green', s=50, label='Start')
    ax.scatter(*traj[-1], color='blue', s=50, label='End')
    ax.scatter(*target, color='red', marker='*', s=150, label='Target')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()
    plt.title('Trajectory with color progression')
    plt.show()

# Configuration
NUM_EPISODES = 10
NEW_TARGET = [10,10,5]

# Load model and evaluate
model = load_latest_model()
trajectories, rewards, distances = evaluate_model(model, NEW_TARGET, NUM_EPISODES)

# Print metrics
print(f"Average reward over {NUM_EPISODES} episodes: {np.mean(rewards):.2f}")
print(f"Average final distance over {NUM_EPISODES} episodes: {np.mean(distances):.2f}")

# Plot a random trajectory
idx = random.randrange(NUM_EPISODES)
print(f"\nPlotting trajectory for episode {idx+1}")
plot_trajectory(trajectories[idx], NEW_TARGET)