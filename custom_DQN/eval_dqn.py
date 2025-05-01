import torch
import numpy as np
from environment import PointMassEnv
from dqn_network import QNetwork
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def evaluate_agent(model, env, n_eval_episodes=40):
    """
    Valuta il modello su n_eval_episodes episodi senza esplorazione.
    Il modello viene messo in modalità evaluation.

    Parametri:
      - model: il modello PyTorch (rete che predice i Q-values)
      - env: l'environment (ad es. PointMassEnv)
      - n_eval_episodes: numero di episodi per la valutazione

    Ritorna:
      - la reward media degli episodi di valutazione.
    """
    model.eval()  # Mette il modello in modalità evaluation
    rewards = []

    for episode in range(n_eval_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # Converti lo stato in tensore e aggiungi una dimensione batch
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = model(state_tensor)  # Predice i Q-values
            # Scegli l'azione con Q-value massimo
            action = q_values.argmax(dim=1).item()

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        rewards.append(total_reward)

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"Reward medio: {mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward

def run_episode_and_plot(env):
    # Resetta l'ambiente e inizia a salvare la traiettoria
    state, _ = env.reset()
    trajectory = [state[:3].copy()]  # memorizza solo le coordinate (x, y, z)
    total_reward = 0.0
    
    # Esegui fino a max_steps o finché l'episodio termina
    for t in range(env.max_steps):
        # Usa una policy random (sostituisci con la tua policy se necessario)
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        trajectory.append(next_state[:3].copy())
        print(f"Step {t+1}: Action: {action} -> Reward: {reward:.2f} | State: {next_state}")
        state = next_state
        if terminated or truncated:
            if terminated:
                print("Target raggiunto.")
            break
    
    print(f"\nTotal reward for the episode: {total_reward:.2f}")
    trajectory = np.array(trajectory)

    # Plot 3D della traiettoria
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label="Traiettoria", color="blue")
    
    # Plotta il target (assunto definito in env.target)
    target = env.target
    ax.scatter(target[0], target[1], target[2], c="red", marker="*", s=200, label="Target")
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Traiettoria del Punto Materiale")
    ax.legend()
    plt.show()

# TODO: Richiedere da CLI nome del modello da valutare
if __name__ == '__main__':
    model_path = "dqn_agent_final.pth" # CAMBIA NOME PRIMA DI ESEGUIRE
    model = QNetwork(input_dim=5, output_dim=6)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    env = PointMassEnv(render_mode="human")
    evaluate_agent(model, env, n_eval_episodes=10)
    run_episode_and_plot(env)