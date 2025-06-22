import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting

from stable_baselines3 import PPO
from environment import PointMassEnv

def load_latest_model(model_dir='models', prefix='ppo_sb3_', ext='.zip'):
    """
    Cerca l'ultimo modello salvato nella directory 'models' e lo carica.
    """
    pattern = os.path.join(model_dir, f"{prefix}*{ext}")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No model files matching {pattern}")
    latest = max(files, key=os.path.getmtime)
    print(f"Loading model: {latest}")
    return PPO.load(latest)


def evaluate_model(model, target, num_episodes=10):
    """
    Valuta il modello per num_episodes episodi.
    Ritorna:
      - trajectories: lista di array shape (T,3) con posizioni successive
      - rewards: lista di reward totali
      - distances: lista di distanze finali
      - target: coordinate target
    """
    env = PointMassEnv()
    trajectories = []
    rewards = []
    distances = []
    env.target = target.copy()

    for ep in range(num_episodes):
        obs_full, _ = env.reset()
        if obs_full.shape[0] > model.observation_space.shape[0]:
            obs = obs_full[: model.observation_space.shape[0]]
        else:
            obs = obs_full
        done = False
        traj = []
        ep_reward = 0.0
        while not done:
            # registra la posizione corrente
            traj.append(env.state[:3].copy())
            # calcola azione deterministica
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        # salva risultati episodio
        trajectories.append(np.array(traj))
        rewards.append(ep_reward)
        final_dist = info.get('distance', np.linalg.norm(env.state[:3] - target))
        distances.append(final_dist)

    return trajectories, rewards, distances, target


def plot_trajectory(traj, target):
    """
    Disegna una traiettoria 3D colorata in base al passo, con il target evidenziato.
    """
    n = len(traj)
    colors = cm.viridis(np.linspace(0, 1, n))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # line segment by segment per avere sfumatura
    for i in range(n-1):
        seg = traj[i:i+2]
        ax.plot(seg[:,0], seg[:,1], seg[:,2], color=colors[i])
    # punti inizio/fine
    ax.scatter(traj[0,0], traj[0,1], traj[0,2], color='green', marker='o', label='Start')
    ax.scatter(traj[-1,0], traj[-1,1], traj[-1,2], color='blue', marker='o', label='End')
    # target
    ax.scatter(target[0], target[1], target[2], color='red', marker='*', s=100, label='Target')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title('Trajectory with color progression')
    plt.show()


if __name__ == '__main__':
    # numero di episodi di valutazione
    NUM_EPISODES = 50

    # carica il modello
    model = load_latest_model()
    target = np.array([7.0, 3.0, 9.0], dtype=np.float32)

    # esegui valutazione
    trajectories, rewards, distances, target = evaluate_model(model, target, NUM_EPISODES)

    #print(f"Average reward over {NUM_EPISODES} episodes: {np.mean(rewards):.2f}")
    print(f"Average final distance over {NUM_EPISODES} episodes: {np.mean(distances):.2f}")

    # seleziona un episodio a caso per il plot
    idx = random.randrange(NUM_EPISODES)
    print(f"\nPlotting trajectory for episode {idx+1}")
    plot_trajectory(trajectories[idx], target)