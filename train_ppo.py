import os
import datetime
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from environment import PointMassEnv
from ppo_network import ActorCritic

def collect_trajectories(env, actor_critic, device, rollout_length):
    """
    Raccoglie un rollout (traiettorie) dall'ambiente con la politica corrente.

    Parameters
    ----------
    env : gym.Env
        Ambiente PointMassEnv.
    actor_critic : ActorCritic
        Modello Actor-Critic.
    device : torch.device
        Dispositivo (GPU o CPU) per l'esecuzione.
    rollout_length : int
        Lunghezza del rollout.

    Returns
    -------
    storage : list
        Lista di dizionari contenenti informazioni per ogni step
    """
    storage = []
    obs = env.reset()
    state = obs[0]
    steps_collected = 0
    while steps_collected < rollout_length:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        mean, std, value = actor_critic(state_tensor)
        # Crea una distribuzione categorica per le azioni
        dist = torch.distributions.Normal(mean, std)
        action_tensor = dist.rsample()                        # reparameterization trick
        log_prob = dist.log_prob(action_tensor).sum(dim=-1).item()   # somma sui tre parametri
        action = action_tensor.detach().cpu().numpy().flatten()        # array di 3 float
        next_obs, rewards, dones, infos = env.step(action.reshape(1, -1))
        next_state = next_obs[0]
        reward = rewards[0]
        done = bool(dones[0])

        storage.append({
            'state': state,
            'next_state': next_state,
            'action': action,
            'reward': reward,
            'done': done,
            'log_prob': log_prob,
            'value': value.item()
        })
        steps_collected += 1
        state = next_state
        if done:
            obs, _ = env.reset()
            state = obs[0]
            
    return storage

def compute_ret_adv(storage, gamma):
    """
    Calcola i returns e le advantage (return - value) per un rollout di states e azioni.

    Parameters
    ----------
    storage : list
        Lista di dizionari contenenti informazioni per ogni step
    gamma : float
        Fattore di sconto per la reward

    Returns
    -------
    returns : list
        Lista dei returns per ogni step
    advantages : list
        Lista delle advantage per ogni step
    """
    returns = []
    advantages = []
    cumulative_return = 0
    # Quando si verifica done, azzeriamo il return cumulativo
    for step in reversed(storage):
        if step['done']:
            cumulative_return = 0
        cumulative_return = step['reward'] + gamma * cumulative_return
        returns.insert(0, cumulative_return)
        advantages.insert(0, cumulative_return - step['value'])
    return returns, advantages

def compute_gae(storage, actor_critic, device, gamma=0.99, lam=0.95):
    """
    Calcola Returns e Advantage con Generalized Advantage Estimation (GAE).
    storage: lista di dict con 'reward', 'value', 'done', e 'state'.
    """
    # Estrai i valori V(s_t)
    values = [step['value'] for step in storage]
    # calcola V(s_{T+1}) sull'ultimo stato non ancora in storage
    last_state = torch.tensor(storage[-1]['next_state'], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        _, _, last_value = actor_critic(last_state)
    values.append(last_value.item())

    advantages = []
    gae = 0.0
    # backward pass
    for t in reversed(range(len(storage))):
        mask = 0.0 if storage[t]['done'] else 1.0
        delta = storage[t]['reward'] + gamma * values[t+1] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages.insert(0, gae)

    # returns = advantage + value
    returns = [adv + v for adv, v in zip(advantages, values[:-1])]
    return returns, advantages

# Aggiornamento del modello tramite la PPO-clip
def update_policy(actor_critic, optimizer, states, actions, old_log_probs, returns, advantages,
                  clip_param=0.2, value_coeff=0.5, entropy_coeff=0.01, ppo_epochs=10, mini_batch_size=64):
    """
    Updates the policy using the Proximal Policy Optimization (PPO) algorithm.

    Returns
    -------
    float
        The total loss accumulated during the PPO update.
    """
    dataset = TensorDataset(states, actions, old_log_probs, returns, advantages)
    loader = DataLoader(dataset, batch_size=mini_batch_size, shuffle=True) # Uso shuffle per migliorare generalizzazione
    total_loss = 0.0

    for _ in range(ppo_epochs):
        for batch in loader:
            state, action, old_lp, returns, advantages = batch
            mean, std, value = actor_critic(state)
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(action).sum(dim=-1)
            ratio = torch.exp(new_log_probs - old_lp)

            # PPO Loss: MIN tra il termine base e quello clippato
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * advantages
            policy_loss = -torch.min(surrogate1, surrogate2).mean()
            # Critic Loss: MSE tra i returns e il valore stimato
            value_loss = ((returns - value.squeeze()) ** 2).mean()
            # Entropy Loss: incentiva l'esplorazione
            entropy_loss = -dist.entropy().mean()

            # Combino le loss
            loss = policy_loss + value_coeff * value_loss + entropy_coeff * entropy_loss
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), max_norm=0.5)

            optimizer.step()
            total_loss += loss.item()
    return total_loss

def train_and_plot(input_dim=8, num_actions=3, total_iterations=100, rollout_length=2048, ppo_epochs=10, mini_batch_size=64, 
                       gamma=0.95, learning_rate=0.0005, clip_param=0.2, entropy_coeff=0.01):
    """
    Addestra un agente PPO e ne plotta i risultati.
    
    Parameters
    ----------
    total_iterations : int
        Numero di iterazioni di addestramento.
    rollout_length : int
        Numero di step per ogni rollout.
    ppo_epochs : int
        Numero di epoche di aggiornamento della policy.
    mini_batch_size : int
        Dimensione del batch per l'aggiornamento della policy.
    gamma : float
        Fattore di sconto per le reward.
    learning_rate : float
        Tasso di apprendimento per l'ottimizzatore.
    clip_param : float
        Parametro di clipping per la PPO.
    
    Returns
    -------
    model : nn.Module
        Il modello del PPO addestrato.
    env : gym.Env
        L'ambiente PointMassEnv utilizzato per l'addestramento.
    episode_rewards : list
        La lista delle reward per episodio.
    episode_final_distances : list
        La lista delle distanze finali dal target per episodio.
    """
    # Inizializza l'ambiente e registra le statistiche degli episodi
    # VectorEnv + normalization
    venv = DummyVecEnv([lambda: PointMassEnv()])
    env  = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Inizializza il modello actor-critic
    actor_critic = ActorCritic(input_dim=input_dim, num_actions=num_actions).to(device)
    optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    episode_rewards = []
    episode_final_distances = []
    all_losses = []
    
    for iteration in range(total_iterations):
        # Raccogli un rollout
        storage = collect_trajectories(env, actor_critic, device, rollout_length)
        returns, advantages = compute_gae(storage, actor_critic, device, gamma=0.99, lam=0.95)
        
        # Converti i dati raccolti in tensori
        states_np = np.array([step['state'] for step in storage], dtype=np.float32)
        states = torch.tensor(states_np, dtype=torch.float32).to(device)
        actions_np = np.array([step['action'] for step in storage], dtype=np.float32)
        actions = torch.tensor(actions_np, dtype=torch.float32).to(device)
        old_log_probs = torch.tensor([step['log_prob'] for step in storage], dtype=torch.float32).to(device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32).to(device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(device)
        # Normalizza gli advantage per una migliore stabilità numerica
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        loss = update_policy(actor_critic, optimizer, states, actions, old_log_probs, returns_tensor, advantages_tensor,
                             clip_param=clip_param, ppo_epochs=ppo_epochs, mini_batch_size=mini_batch_size)
        all_losses.append(loss)
        
        # Per il monitoraggio, esegui un episodio di test in modalità greedy
        raw_env = env.venv.envs[0]
        raw_state, _ = raw_env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            norm_state = env.normalize_obs(raw_state)
            with torch.no_grad():
                mean, std, _ = actor_critic(torch.tensor(norm_state, dtype=torch.float32).unsqueeze(0).to(device))
            action = mean.squeeze(0).cpu().numpy()
            raw_state, reward, terminated, truncated, info = raw_env.step(action)
            episode_reward += reward
            done = bool(terminated or truncated)
        final_distance = np.linalg.norm(raw_state[:3] - raw_env.target)
        episode_rewards.append(episode_reward)
        episode_final_distances.append(final_distance)
        
        print(f"Iteration {iteration+1}/{total_iterations} | Reward: {episode_reward:.2f} | Final Distance: {final_distance:.2f}")
    
    # Plot dei risultati
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
    
    # Plot dell'andamento della loss
    plt.figure(figsize=(8,6))
    plt.plot(all_losses, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("PPO Loss")
    plt.title("PPO Loss over Iterations")
    plt.show()
    
    # Salva il modello
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"ppo_agent_{timestamp}.pth")
    torch.save(actor_critic.state_dict(), model_path)
    
    print("\nTraining completed!")
    return actor_critic, env, episode_rewards, episode_final_distances

# Parametri della rete
input_dim = 11
num_actions = 3
total_iterations = 500
rollout_length = 128
ppo_epochs = 20
mini_batch_size = 32
learning_rate = 0.0001
entropy_coeff = 0.02
clip_param = 0.2

actor_critic, env, rewards_log, distances_log = train_and_plot(input_dim=input_dim,
                                                               num_actions=num_actions,
                                                               total_iterations=total_iterations,
                                                               rollout_length=rollout_length,
                                                               ppo_epochs=ppo_epochs,
                                                               mini_batch_size=mini_batch_size,
                                                               learning_rate=learning_rate,
                                                               entropy_coeff=entropy_coeff,
                                                               clip_param=clip_param)