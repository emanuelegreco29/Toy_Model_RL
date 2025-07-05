import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from stable_baselines3 import PPO
from train.dogfight_sb3 import make_env, MultiAgentWrapper  # :contentReference[oaicite:6]{index=6}
from environments.dogfight_env import DogfightParallelEnv  # :contentReference[oaicite:7]{index=7}

def load_latest_model(model_dir='models', prefix='ppo_selfplay_', ext='.zip'):
    files = glob.glob(os.path.join(model_dir, f"{prefix}*{ext}"))
    if not files:
        raise FileNotFoundError(f"No model files matching {prefix}*{ext}")
    latest = max(files, key=os.path.getmtime)
    print(f"Loading model: {latest}")
    return PPO.load(latest, device='cpu')

def evaluate_model(model, num_episodes=10):
    traj_agent, traj_opp, rewards, final_hps, hit_idxs = [], [], [], [], []
    # crea wrapper e imposta l'opponente come la stessa policy
    env = make_env()  # :contentReference[oaicite:8]{index=8}
    env.set_opponent_policy(model)
    for ep in range(num_episodes):
        obs, _ = env.reset()
        agent_path, opp_path = [], []
        ep_reward = 0.0
        hp_prev = None
        hits = []
        step = 0
        while True:
            # stato interno (x,y,z,v,yaw,pitch) per entrambi
            states = env.env.states
            a_st = states[env.controlled_agent]
            o_st = states[env.opponent_agent]
            agent_path.append(a_st.copy())
            opp_path.append(o_st.copy())

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward

            # traccia colpi: hp dell'avversario
            hp_curr = env.env.hp[env.opponent_agent]
            if hp_prev is None:
                hp_prev = hp_curr
            if hp_curr < hp_prev:
                hits.append(step)
            hp_prev = hp_curr
            step += 1

            if done or truncated:
                # ultimo stato
                states = env.env.states
                agent_path.append(states[env.controlled_agent].copy())
                opp_path.append(states[env.opponent_agent].copy())

                traj_agent.append(np.array(agent_path))
                traj_opp.append(np.array(opp_path))
                rewards.append(ep_reward)
                final_hps.append(env.env.hp[env.opponent_agent])
                hit_idxs.append(hits)
                break

    return traj_agent, traj_opp, rewards, final_hps, hit_idxs

def plot_coordinates(agent_traj, opp_traj, hit_indices, title, save_dir):
    coords = ['X','Y','Z']
    for i, c in enumerate(coords):
        plt.figure()
        plt.plot(agent_traj[:,i], label='Agent')
        plt.plot(opp_traj[:,i], label='Opponent', linestyle='--')
        if hit_indices:
            plt.scatter(hit_indices, opp_traj[hit_indices,i], marker='x', label='Hit')
        plt.xlabel('Timestep'); plt.ylabel(f'{c}(t)')
        plt.title(f'{title} – {c}(t)')
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{title}_{c}.png'))
        plt.close()

def animate_with_wez(agent_traj, opp_traj, filename, save_dir, pad=1.0):
    env_params = DogfightParallelEnv()
    wez_length = env_params.wez_length
    wez_angle  = env_params.wez_angle

    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    a_line, = ax.plot([],[],[], lw=2, color='blue',  label='Agent')
    o_line, = ax.plot([],[],[], lw=2, color='orange', linestyle='-', label='Opponent')
    b1,     = ax.plot([],[],[], lw=1, color='green', alpha=0.5)
    b2,     = ax.plot([],[],[], lw=1, color='green', alpha=0.5)

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z'); ax.legend()

    def update(i):
        a = agent_traj[i]; o = opp_traj[i]
        # traiettorie
        a_line.set_data(agent_traj[:i+1,0], agent_traj[:i+1,1])
        a_line.set_3d_properties(agent_traj[:i+1,2])
        o_line.set_data(opp_traj[:i+1,0], opp_traj[:i+1,1])
        o_line.set_3d_properties(opp_traj[:i+1,2])
        # WEZ edges
        yaw, pitch = a[4], a[5]
        left  = np.array([np.cos(pitch)*np.cos(yaw+wez_angle),
                          np.cos(pitch)*np.sin(yaw+wez_angle),
                          np.sin(pitch)])
        right = np.array([np.cos(pitch)*np.cos(yaw-wez_angle),
                          np.cos(pitch)*np.sin(yaw-wez_angle),
                          np.sin(pitch)])
        p1 = a[:3] + left * wez_length
        p2 = a[:3] + right* wez_length
        b1.set_data([a[0],p1[0]], [a[1],p1[1]]); b1.set_3d_properties([a[2],p1[2]])
        b2.set_data([a[0],p2[0]], [a[1],p2[1]]); b2.set_3d_properties([a[2],p2[2]])
        # limiti
        mins = np.minimum(a[:3], o[:3]) - pad
        maxs = np.maximum(a[:3], o[:3]) + pad
        ax.set_xlim(mins[0],maxs[0]); ax.set_ylim(mins[1],maxs[1]); ax.set_zlim(mins[2],maxs[2])
        return a_line, o_line, b1, b2

    path = os.path.join(save_dir, filename)
    ani = FuncAnimation(fig, update, frames=len(agent_traj), interval=200, blit=False)
    ani.save(path, writer=PillowWriter(fps=5))
    plt.close()

if __name__ == "__main__":
    model = load_latest_model()
    print("Model loaded.")
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M')
    save_dir = os.path.join('plots', f"dogfight_eval_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    num_ep = 100
    print(f"Evaluating for {num_ep} episodes…")
    traj_a, traj_o, rewards, hps, hits = evaluate_model(model, num_ep)

    # seleziona best/worst per HP finale (min hp → più kill)
    min_hp = np.min(hps); max_hp = np.max(hps)
    best = np.argmin(hps)
    worst= np.argmax(hps)

    print(f"Best Ep {best+1} | Final HP: {hps[best]} | Reward: {rewards[best]:.2f}")
    print(f"Worst Ep{worst+1} | Final HP: {hps[worst]} | Reward: {rewards[worst]:.2f}")

    # plot
    print("Plotting coordinates…")
    plot_coordinates(traj_a[best], traj_o[best], hits[best],  'Best',  save_dir)
    plot_coordinates(traj_a[worst], traj_o[worst], hits[worst],'Worst', save_dir)

    # animazioni
    print("Creating animations…")
    animate_with_wez(traj_a[best], traj_o[best],   'best.gif',  save_dir)
    animate_with_wez(traj_a[worst],traj_o[worst],  'worst.gif', save_dir)

    print(f"Results saved to {save_dir}")
