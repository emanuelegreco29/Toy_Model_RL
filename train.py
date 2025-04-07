import random
import os
import numpy as np
import torch
import torch.optim as optim
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

from gymnasium.wrappers import RecordEpisodeStatistics

from environment import PointMassEnv
from network import QNetwork, ReplayBuffer, huber_loss

action_map = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
    4: "SPEED",
    5: "BREAK"
}

def train_and_plot(obs_dim=5, action_dim=6, learning_rate=0.001, gamma=0.95, buffer_capacity=100000, batch_size=128, total_steps=100000, target_update_interval=500, train_freq=4, epsilon_decay=0.5):
    """
    Esegue l'addestramento di un agente Deep Q-Network per l'ambiente PointMassEnv.
    
    Parameters
    ----------
    obs_dim : int
        Dimensione dell'osservazione dell'agente.
    action_dim : int
        Numero di azioni possibili.
    learning_rate : float
        Tasso di apprendimento per l'ottimizzatore.
    gamma : float
        Fattore di sconto per la reward.
    buffer_capacity : int
        Capacit  del buffer di esperienze.
    batch_size : int
        Numero di esperienze campionate per ogni ottimizzazione.
    total_steps : int
        Numero di step massimo per l'addestramento.
    target_update_interval : int
        Numero di step tra gli aggiornamenti del target network.
    train_freq : int
        Numero di step tra le ottimizzazioni.
    epsilon_decay : float
        Frazione di total_steps durante cui ε decresce da 1.0 a 0.05.
    
    Returns
    -------
    model : nn.Module
        Il modello del Deep Q-Network addestrato.
    env : gym.Env
        L'ambiente PointMassEnv utilizzato per l'addestramento.
    episode_rewards : list
        La lista delle reward per episodio.
    episode_final_distances : list
        La lista delle distanze finali dal target per episodio.
    """
    
    # Inizializza ambiente e registra le statistiche degli episodi
    env = PointMassEnv(render_mode="human")
    env = RecordEpisodeStatistics(env)
    
    # Usa GPU se disponibile
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Inizializza la rete attuale e la rete target
    model = QNetwork(input_dim=obs_dim, output_dim=action_dim).to(device)
    model_target = QNetwork(input_dim=obs_dim, output_dim=action_dim).to(device)
    model_target.load_state_dict(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)
    
    # Parametri per l'epsilon-greedy
    epsilon_start = 1.0
    epsilon_final = 0.05
    epsilon_decay = epsilon_decay  # frazione di total_steps durante cui ε decresce da start a final
    epsilon_by_timestep = lambda t: epsilon_final + (epsilon_start - epsilon_final) * np.exp(-t / (total_steps * epsilon_decay))
    
    state, _ = env.reset()
    episode_reward = 0.0
    episode_count = 0
    
    # Liste per loggare i risultati per episodio
    episode_rewards = []
    episode_final_distances = []

    # File per il log degli episodi
    log_dir = "logs"
    log_path = os.path.join(log_dir, f"training_log_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt")
    log_file = open(log_path, "w")

    log_file.write(f"Episode {episode_count}\n\n")
    step_in_episode = 0  # Contatore per lo step all'interno dell'episodio
    
    for t in range(1, total_steps+1):
        step_in_episode += 1
        epsilon = epsilon_by_timestep(t)
        # Azione ε‑greedy
        if random.random() < epsilon:
            action = env.action_space.sample()
            # Per logs
            q_value = None
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                q_vals = model(state_tensor)
            action = q_vals.argmax(dim=1).item()
            # Per logs
            q_value = q_vals[0, action].item()
        
        next_state, reward, terminated, truncated, _ = env.step(action)

        # Per logs
        action_label = action_map.get(action, str(action))
        # Formatta lo stato: arrotonda ogni valore a 2 cifre decimali
        formatted_state = "[" + ", ".join(f"{s:.2f}" for s in state.tolist()) + "]"
        formatted_q_value = f"{q_value:.2f}" if q_value is not None else "None"
        formatted_reward = f"{reward:.2f}"
        log_line = (f"{step_in_episode}) State: {formatted_state} | Action: {action_label} | "
                f"Q_value: {formatted_q_value} | Reward: {formatted_reward}\n")
        log_file.write(log_line)

        # Aggiorna il contenuto del file di log
        if t % 1000 == 0:
            log_file.flush()

        done = terminated or truncated
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        
        if done:
            episode_count += 1
            step_in_episode = 0
            log_file.write(f"\nEpisode {episode_count}\n\n")
            final_distance = np.linalg.norm(state[:3] - env.unwrapped.target)
            episode_rewards.append(episode_reward)
            episode_final_distances.append(final_distance)
            print(f"Episode: {episode_count} | Reward: {episode_reward:.2f} | Final Distance: {final_distance:.2f} | Epsilon: {epsilon:.2f}")
            state, _ = env.reset()
            episode_reward = 0.0
        
        if len(replay_buffer) >= batch_size and t % train_freq == 0:
            batch = replay_buffer.sample(batch_size)
            obs_batch = batch['obs'].to(device)
            actions_batch = batch['actions'].to(device)
            rewards_batch = batch['reward'].to(device)
            next_state_batch = batch['next_state'].to(device)
            dones_batch = batch['done'].to(device)
            
            q_values = model(obs_batch).gather(1, actions_batch)
            with torch.no_grad():
                # Double DQN: il modello online seleziona l'azione migliore per il next state
                next_actions = model(next_state_batch).argmax(dim=1, keepdim=True)
                # Il target network valuta l'azione selezionata
                next_q_values = model_target(next_state_batch).gather(1, next_actions)
            target_q_values = rewards_batch + (0.95 * next_q_values * (1 - dones_batch))

            loss = huber_loss(q_values, target_q_values)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if t % target_update_interval == 0:
                model_target.load_state_dict(model.state_dict())
                #print(f"Step: {t} | Epsilon: {epsilon:.2f} | Loss: {loss.item():.4f}")
    
    # Plotta l'evoluzione della reward e della distanza finale per episodio
    episodes = np.arange(1, len(episode_rewards) + 1)
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    axs[0].plot(episodes, episode_rewards, marker='o')
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Total Reward")
    axs[0].set_title("Total Reward per Episode")
    
    axs[1].plot(episodes, episode_final_distances, marker='o', color='orange')
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Final Distance")
    axs[1].set_title("Final Distance from Target per Episode")
    plt.tight_layout()
    plt.show()
    
    # Esegui un episodio di test (greedy) per registrare la traiettoria
    trajectory = []
    state, _ = env.reset()
    trajectory.append(state[:3].copy())
    done = False
    while not done:
        model.eval()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_vals = model(state_tensor)
        model.train()
        action = q_vals.argmax(dim=1).item()
        next_state, reward, terminated, truncated, _ = env.step(action)
        trajectory.append(next_state[:3].copy())
        state = next_state
        done = terminated or truncated

    trajectory = np.array(trajectory)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Calcola il gradiente basato sul numero di punti nella traiettoria
    num_points = len(trajectory)
    norm = plt.Normalize(0, num_points - 1)
    cmap = cm.viridis

    # Traccia segmento per segmento con il colore che varia in base all'indice
    for i in range(num_points - 1):
        seg = trajectory[i:i+2]
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=cmap(norm(i)))

    # Evidenzia il punto di partenza e di arrivo
    ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], color='green', s=100, label='Start')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], color='red', s=100, label='End')

    # Plotta il target
    ax.scatter(env.unwrapped.target[0], env.unwrapped.target[1], env.unwrapped.target[2], color='blue', marker="*", s=200, label="Target")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Trajectory of the Trained Agent (Test Episode) with Gradient")
    ax.legend()
    plt.show()

    
    # Salva il modello
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = "models"
    model_path = os.path.join(model_dir, f"dqn_agent_{timestamp}.pth")
    torch.save(model.state_dict(), model_path)
    log_file.close()
    print("\nTraining completed!")
    
    return model, env, episode_rewards, episode_final_distances


# Parametri della rete
obs_dim = 5
action_dim = 6
learning_rate = 0.0005
gamma = 0.95
buffer_capacity = 100000
batch_size = 128
total_steps = 500000
target_update_interval = 1000
epsilon_decay = 0.85

model, env, rewards_log, distances_log = train_and_plot(
    obs_dim=obs_dim,
    action_dim=action_dim,
    learning_rate=learning_rate,
    gamma=gamma,
    buffer_capacity=buffer_capacity,
    batch_size=batch_size,
    total_steps=total_steps,
    target_update_interval=target_update_interval,
    train_freq=4,
    epsilon_decay=epsilon_decay
)