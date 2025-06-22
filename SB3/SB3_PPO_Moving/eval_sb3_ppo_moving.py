import os, glob, random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import PPO
from env_sb3_ppo_moving import PointMassEnv

def load_latest_model(model_dir='models', prefix='ppo_sb3_moving_target_', ext='.zip'):
    files = glob.glob(os.path.join(model_dir, f"{prefix}*{ext}"))
    if not files:
        raise FileNotFoundError(f"No model files matching {prefix}*{ext}")
    latest = max(files, key=os.path.getmtime)
    print(f"Loading model: {latest}")
    return PPO.load(latest)

def evaluate_model(model, num_episodes=10):
    trajectories, target_trajs, rewards, distances = [], [], [], []
    env = PointMassEnv()
    for _ in range(num_episodes):
        obs, _ = env.reset()
        agent_path, target_path = [], []
        ep_reward = 0.0
        while True:
            agent_path.append(env.state[:3].copy())
            target_path.append(env.target_state[:3].copy())
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                agent_path.append(env.state[:3].copy())
                target_path.append(env.target_state[:3].copy())
                rewards.append(ep_reward)
                distances.append(info.get('distance', np.nan))
                break
        trajectories.append(np.array(agent_path))
        target_trajs.append(np.array(target_path))
    return trajectories, target_trajs, rewards, distances

def plot_trajectory(agent_traj, target_traj, title='Trajectory'):
    n = len(agent_traj)
    colors = cm.viridis(np.linspace(0,1,n))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(n-1):
        seg = agent_traj[i:i+2]
        ax.plot(seg[:,0], seg[:,1], seg[:,2], color=colors[i])
    ax.plot(target_traj[:,0], target_traj[:,1], target_traj[:,2], '--', color='red', label='Target path')
    ax.scatter(*agent_traj[0], color='green', s=50, label='Agent start')
    ax.scatter(*agent_traj[-1], color='blue', s=50, label='Agent end')
    ax.scatter(*target_traj[0], color='orange', marker='*', s=80, label='Target start')
    ax.scatter(*target_traj[-1], color='red', marker='*', s=80, label='Target end')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend(); plt.title(title); plt.show()

# Load and evaluate
model = load_latest_model()
num_ep = 100
trajectories, target_trajs, rewards, distances = evaluate_model(model, num_ep)

print(f"Average reward: {np.mean(rewards):.2f} | Average distance: {np.mean(distances):.2f}")

# Identify best and worst episodes by reward
best_idx = int(np.argmax(rewards))
worst_idx = int(np.argmin(rewards))
print(f"Best Episode {best_idx+1} | Reward: {rewards[best_idx]:.2f} | Distance: {distances[best_idx]:.2f}")
print(f"Worst Episode {worst_idx+1} | Reward: {rewards[worst_idx]:.2f} | Distance: {distances[worst_idx]:.2f}")

# Plot best and worst trajectories
plot_trajectory(trajectories[best_idx], target_trajs[best_idx], title='Best Episode')
plot_trajectory(trajectories[worst_idx], target_trajs[worst_idx], title='Worst Episode')