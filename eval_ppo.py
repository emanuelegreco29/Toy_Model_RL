import torch
import os
import numpy as np
from environment import PointMassEnv
from ppo_network import ActorCritic
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

action_map = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
    8: "UP+RIGHT",
    5: "UP+LEFT",
    6: "DOWN+RIGHT",
    7: "DOWN+LEFT",
    4: "HOLD"
}

def evaluate_agent(model, env, n_episodes=10):
    """
    Valuta il modello PPO per n_episodes episodi in modalità deterministica.
    Il modello viene messo in modalità evaluation.
    
    Ritorna la reward media e stampa la media e la deviazione standard.
    """
    model.eval()
    rewards = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model(state_tensor)  # Ricava i logits per la politica e il value
            # Seleziona l'azione con massima probabilità (argmax)
            action = torch.argmax(logits, dim=1).item()

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        rewards.append(total_reward)

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"Reward medio: {mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward

def run_plot(env, model):
    """
    Esegue un episodio utilizzando la policy derivata dal modello PPO e plotta la traiettoria 3D.
    """
    state, _ = env.reset()
    trajectory = [state[:3].copy()]
    total_reward = 0.0
    done = False
    t = 0

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(state_tensor)
        # Seleziona l'azione con massima probabilità (argmax)
        action = torch.argmax(logits, dim=1).item()
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        # Formatta lo stato con arrotondamento a 2 cifre decimali
        formatted_state = "[" + ", ".join(f"{s:.2f}" for s in state.tolist()) + "]"
        # Mappa l'azione usando l'action_map
        action_label = action_map.get(action, str(action))

        trajectory.append(next_state[:3].copy())
        print(f"Step {t+1}) Reward: {reward:.2f} | State: {formatted_state} | Action: {action_label}")
        state = next_state
        done = terminated or truncated
        t += 1

    print(f"\nTotal reward for the episode: {total_reward:.2f}")
    trajectory = np.array(trajectory)

    # Plot 3D della traiettoria
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label="Traiettoria", color="blue")
    
    # Plotta il target
    target = env.target
    ax.scatter(target[0], target[1], target[2], color='red', marker="*", s=200, label="Target")
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Traiettoria del Punto Materiale")
    ax.legend()
    plt.show()


model_dir = "models"
model_name = "ppo_agent_20250414-170310.pth"  # INSERIRE NOME FILE PRIMA DI ESEGUIRE
model_path = os.path.join(model_dir, model_name)
# Inizializza il modello PPO
model = ActorCritic(input_dim=8, num_actions=9)
# Carica i pesi del modello
model.load_state_dict(torch.load(model_path))
model.eval()
env = PointMassEnv(render_mode="human")
    
# Valutazione dell'agente
evaluate_agent(model, env, n_episodes=50)
print("\n\n")
run_plot(env, model)