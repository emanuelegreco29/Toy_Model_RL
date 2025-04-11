import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import RecordEpisodeStatistics
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import datetime

class PointMassEnv(gym.Env):
    """
    Ambiente semplificato per un punto materiale in 3D.
    Stato: [x, y, z, v, theta]
      - (x, y, z): posizione
      - v: velocità scalare
      - theta: orientamento nel piano xy (radianti)
    Azioni:
      0: Salire (incrementa z)
      1: Scendere (decrementa z)
      2: Girare a destra (incrementa theta)
      3: Girare a sinistra (decrementa theta)
      4: Accelerare (incrementa v)
      5: Frenare (diminuisce v)
    """
    def __init__(self, render_mode=None):
        super(PointMassEnv, self).__init__()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.dt = 0.1
        self.max_steps = 400
        self.current_step = 0
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.last_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Parametri per aggiornare lo stato
        self.delta_v = 0.25      # incremento velocità
        self.delta_theta = 0.15  # incremento angolo (radianti)
        self.delta_z = 0.25      # incremento quota
        self.v_max = 15       # velocità massima

        # Target fisso (da poter modificare in seguito)
        self.target = np.array([10.0, 10.0, 5.0])
        self.last_distance = np.linalg.norm(self.state[:3] - self.target)

        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.last_distance = np.linalg.norm(self.state[:3] - self.target)
        return self.state.copy(), {}

    def _get_obs(self):
        rel_pos = self.target - self.state[:3]
        return np.concatenate([rel_pos, [self.state[3], self.state[4]]]).astype(np.float32)

    def compute_reward(self, prev_state, current_state, target):
        # Estrai le posizioni 3D dagli stati
        prev_pos = prev_state[:3]
        curr_pos = current_state[:3]

        # Calcola le distanze dal target
        prev_distance = np.linalg.norm(prev_pos - target)
        curr_distance = np.linalg.norm(curr_pos - target)

        # Reward per il progresso: premio proporzionale alla riduzione della distanza
        improvement = prev_distance - curr_distance
        reward = max(improvement * 10.0, 0.0)

        # Bonus
        if curr_distance < 1.0:
            reward += 50.0
        elif curr_distance < 2.0:
            reward += 20.0

        # Bonus per l'allineamento sugli assi
        alignment_bonus = 0.0
        for i in range(3):
            diff = abs(curr_pos[i] - target[i])
            if diff < 0.5:
                alignment_bonus += 1.0

        reward += alignment_bonus
        
        return reward


    def step(self, action):
        x, y, z, v, theta = self.state
        self.last_state = self.state.copy()

        if action == 0:      # Salire
            z += self.delta_z
        elif action == 1:    # Scendere
            z -= self.delta_z
        elif action == 2:    # Girare a destra
            theta += self.delta_theta
        elif action == 3:    # Girare a sinistra
            theta -= self.delta_theta
        elif action == 4:    # Accelerare
            v = min(v + self.delta_v, self.v_max)
        elif action == 5:    # Frenare
            v = max(v - self.delta_v, 0.0)

        # Aggiornamento posizione
        x += v * np.cos(theta) * self.dt
        y += v * np.sin(theta) * self.dt
        self.state = np.array([x, y, z, v, theta], dtype=np.float32)
        self.current_step += 1
        current_distance = np.linalg.norm(self.state[:3] - self.target)

        reward = self.compute_reward(self.last_state, self.state, self.target)

        terminated = False
        truncated = False
        if current_distance < 0.5:
            terminated = True
        elif self.current_step >= self.max_steps:
            truncated = True

        self.last_distance = current_distance
        return self.state.copy(), reward, terminated, truncated, {}

    def render(self, mode='human'):
        if self.verbose:
            print("Distanza dal target:", np.linalg.norm(self.state[:3] - self.target))


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

def custom_dqn_loss(q_values, target_q_values):
    loss_fn = nn.SmoothL1Loss()
    return loss_fn(q_values, target_q_values)

class QNetwork(nn.Module):
    def __init__(self, input_dim=5, output_dim=6):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return {
            'obs': torch.tensor(states, dtype=torch.float32),
            'actions': torch.tensor(actions, dtype=torch.long).unsqueeze(1),
            'reward': torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
            'next_obs': torch.tensor(next_states, dtype=torch.float32),
            'done': torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        }

    def __len__(self):
        return len(self.buffer)

def train_dqn(obs_dim=5, action_dim=6, learning_rate=0.001, gamma=0.95, buffer_capacity=100000, batch_size=128,  total_steps=100000, target_update_interval=500, train_freq=4):
    # Inizializza ambiente e monitor
    env = PointMassEnv(render_mode="human")
    env = RecordEpisodeStatistics(env)

    # Imposta dispositivo
    device = torch.device("cpu")

    # Inizializza reti: la rete attuale e la rete target (copia dei parametri)
    model = QNetwork(input_dim=obs_dim, output_dim=action_dim).to(device)
    model_target = QNetwork(input_dim=obs_dim, output_dim=action_dim).to(device)
    for param in model.parameters():
      param.requires_grad = True
    for param in model_target.parameters():
      param.requires_grad = True
    model.train()
    model_target.train()
    model_target.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    epsilon_start = 1.0
    epsilon_final = 0.05
    epsilon_decay = 0.5 # frazione di total_timesteps durante cui ε scende da start a final

    epsilon_by_timestep = lambda t: epsilon_final + (epsilon_start - epsilon_final) * \
        np.exp(-t / (total_steps * epsilon_decay))

    state, _ = env.reset()
    episode_reward = 0
    episode_count = 0

    for t in range(1, total_steps+1):
        epsilon = epsilon_by_timestep(t)
        # Azione ε‑greedy
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                q_vals = model(state_tensor)
            action = q_vals.argmax(dim=1).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        # Se l'episodio è terminato, resetta e stampa il log
        if done:
            episode_count += 1
            state_str = "[" + ", ".join(f"{s:.1f}" for s in env.unwrapped.state) + "]"
            print(f"Episode: {episode_count} | Reward: {episode_reward:.2f} | State: {state_str} | Epsilon: {epsilon:.2f}")
            state, _ = env.reset()
            episode_reward = 0

        # Aggiornamento della rete: ogni 'train_freq' passi se abbiamo abbastanza esperienze
        if len(replay_buffer) >= batch_size and t % train_freq == 0:
            batch = replay_buffer.sample(batch_size)
            # Sposta i tensori sul device
            obs = batch['obs'].to(device)
            actions = batch['actions'].to(device)
            rewards = batch['reward'].to(device)
            next_obs = batch['next_obs'].to(device)
            dones = batch['done'].to(device)

            # Calcola i Q-values per gli stati correnti per le azioni selezionate
            q_values = model(obs).gather(1, actions)
            # Calcola i target Q-values secondo la formula di Bellman:
            with torch.no_grad():
                next_q_values = model_target(next_obs).max(dim=1)[0].unsqueeze(1)
            # Nota: qui (1 - dones) per evitare di propagare valore se done==True.
            target_q_values = rewards + (0.95 * next_q_values * (1 - dones))

            loss = custom_dqn_loss(q_values, target_q_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Esegui lo scheduler (qui lo facciamo ad ogni update)
            scheduler.step()

            # Aggiorna periodicamente il target network
            if t % target_update_interval == 0:
                model_target.load_state_dict(model.state_dict())
                #print(f"Step: {t} | Epsilon: {epsilon:.2f} | Loss: {loss.item():.4f}")

    # Salva il modello con timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path = f"dqn_agent_{timestamp}.pth"
    torch.save(model.state_dict(), model_path)
    print("\nTraining completed!")
    return model, env

def train_and_plot(obs_dim=5, action_dim=6, learning_rate=0.001, gamma=0.95, 
                   buffer_capacity=100000, batch_size=128, total_steps=100000, 
                   target_update_interval=500, train_freq=4):
    # Inizializza ambiente e registra le statistiche degli episodi
    env = PointMassEnv(render_mode="human")
    env = RecordEpisodeStatistics(env)
    
    device = torch.device("cpu")
    
    # Inizializza la rete attuale e la rete target
    model = QNetwork(input_dim=obs_dim, output_dim=action_dim).to(device)
    model_target = QNetwork(input_dim=obs_dim, output_dim=action_dim).to(device)
    model_target.load_state_dict(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Qui usiamo gamma anche come fattore di decay dello scheduler (puoi modificarlo se vuoi)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)
    
    # Parametri per l'epsilon-greedy
    epsilon_start = 1.0
    epsilon_final = 0.05
    epsilon_decay = 0.3  # frazione di total_steps durante cui ε decresce da start a final
    epsilon_by_timestep = lambda t: epsilon_final + (epsilon_start - epsilon_final) * \
        np.exp(-t / (total_steps * epsilon_decay))
    
    state, _ = env.reset()
    episode_reward = 0.0
    episode_count = 0
    
    # Liste per loggare i risultati per episodio
    episode_rewards = []
    episode_final_distances = []
    
    for t in range(1, total_steps+1):
        epsilon = epsilon_by_timestep(t)
        # Azione ε‑greedy
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                q_vals = model(state_tensor)
            action = q_vals.argmax(dim=1).item()
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        
        if done:
            episode_count += 1
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
            next_obs_batch = batch['next_obs'].to(device)
            dones_batch = batch['done'].to(device)
            
            q_values = model(obs_batch).gather(1, actions_batch)
            with torch.no_grad():
                next_q_values = model_target(next_obs_batch).max(dim=1)[0].unsqueeze(1)
            target_q_values = rewards_batch + (0.95 * next_q_values * (1 - dones_batch))
            
            loss = custom_dqn_loss(q_values, target_q_values)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if t % target_update_interval == 0:
                model_target.load_state_dict(model.state_dict())
                print(f"Step: {t} | Epsilon: {epsilon:.2f} | Loss: {loss.item():.4f}")
    
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
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_vals = model(state_tensor)
        action = q_vals.argmax(dim=1).item()
        next_state, reward, terminated, truncated, _ = env.step(action)
        trajectory.append(next_state[:3].copy())
        state = next_state
        done = terminated or truncated
    
    trajectory = np.array(trajectory)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label="Trajectory", color="blue")
    ax.scatter(env.unwrapped.target[0], env.unwrapped.target[1], env.unwrapped.target[2], c="red", marker="*", s=200, label="Target")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Trajectory of the Trained Agent (Test Episode)")
    ax.legend()
    plt.show()
    
    # Salva il modello
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path = f"dqn_agent_{timestamp}.pth"
    torch.save(model.state_dict(), model_path)
    print("\nTraining completed!")
    
    return model, env, episode_rewards, episode_final_distances

# Parametri della rete
obs_dim = 5
action_dim = 6
learning_rate = 0.001
gamma = 0.95
buffer_capacity = 100000
batch_size = 128
total_steps = 300000
target_update_interval = 500

model, env, rewards_log, distances_log = train_and_plot(obs_dim=obs_dim,
                       action_dim=action_dim,
                       learning_rate=learning_rate,
                       gamma=gamma,
                       buffer_capacity=buffer_capacity,
                       batch_size=batch_size,
                       total_steps=total_steps,
                       target_update_interval=target_update_interval,
                       train_freq=4)

# evaluate_agent(model, env, 70)

# env = PointMassEnv(render_mode="human")
# run_episode_and_plot(env)