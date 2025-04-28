import torch
import os
import numpy as np
from environment import PointMassEnv
from ppo_network import ActorCritic
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def evaluate_agent(model, env, n_episodes=10):
    """
    Valuta il modello PPO per n_episodes episodi in modalità deterministica.
    Restituisce la reward media.
    """
    model.eval()
    rewards = []

    for episode in range(n_episodes):
        # reset raw environment
        state, _ = env.reset()  # ora state è un vettore di dimensione 11
        done = False
        total_reward = 0.0

        while not done:
            # prepariamo il tensore di input
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                mean, std, _ = model(state_tensor)
            # estraiamo l'azione (deterministica)
            action = mean.squeeze(0).cpu().numpy()
            # clipping nell'action_space
            action = np.clip(action, env.action_space.low, env.action_space.high)

            # passo nell'ambiente raw: restituisce next_state, reward, terminated, truncated, info
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward

        rewards.append(total_reward)

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"Average reward over {n_episodes} episodes: {mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward


def run_plot(env, model):
    """
    Esegue un episodio mostrano i dettagli e plotta la traiettoria 3D.
    """
    state, _ = env.reset()
    trajectory = [state[:3].copy()]
    total_reward = 0.0
    done = False
    t = 0

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mean, std, _ = model(state_tensor)
        action = mean.squeeze(0).cpu().numpy()
        action = np.clip(action, env.action_space.low, env.action_space.high)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # formattazione per stampa
        s_fmt = "[" + ", ".join(f"{x:.2f}" for x in state) + "]"
        a_fmt = "[" + ", ".join(f"{x:.2f}" for x in action) + "]"
        print(f"Step {t+1}) Reward: {reward:.2f} | State: {s_fmt} | Action: {a_fmt}")

        trajectory.append(next_state[:3].copy())
        state = next_state
        total_reward += reward
        t += 1

    print(f"\nTotal reward for the episode: {total_reward:.2f}")
    traj = np.array(trajectory)

    # plot 3D
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(traj[:,0], traj[:,1], traj[:,2], label="Trajectory")
    ax.scatter(env.target[0], env.target[1], env.target[2], color='red', marker='*', s=200, label='Target')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('Trajectory of the Point Mass')
    ax.legend()
    plt.show()


model_dir = "models"
model_name = "ppo_agent_20250427-165954.pth"  # aggiorna se necessario
model_path = os.path.join(model_dir, model_name)

# inizializza e carica il modello
model = ActorCritic(input_dim=11, num_actions=3)
model.load_state_dict(torch.load(model_path))
model.eval()

# usa l'ambiente raw per coerenza con training
env = PointMassEnv(render_mode="human")

# valutazione e plot
evaluate_agent(model, env, n_episodes=50)
print("\n\n")
run_plot(env, model)
