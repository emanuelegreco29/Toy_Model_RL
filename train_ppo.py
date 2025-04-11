import os
import datetime
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from gymnasium.wrappers import RecordEpisodeStatistics

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
    state, _ = env.reset()
    steps_collected = 0
    while steps_collected < rollout_length:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        logits, value = actor_critic(state_tensor)
        # Crea una distribuzione categorica per le azioni
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        storage.append({
            'state': state,
            'action': action.item(),
            'reward': reward,
            'done': done,
            'log_prob': log_prob.item(),
            'value': value.item()
        })
        steps_collected += 1
        state = next_state
        if done:
            if terminated:
                print("Target reached!")
            state, _ = env.reset()
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
            logits, value = actor_critic(state)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(action)
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
            optimizer.step()
            total_loss += loss.item()
    return total_loss

def train_and_plot(input_dim=8, num_actions=9, total_iterations=100, rollout_length=2048, ppo_epochs=10, mini_batch_size=64, 
                       gamma=0.95, learning_rate=0.0005, clip_param=0.2):
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
    env = PointMassEnv(render_mode="human")
    env = RecordEpisodeStatistics(env)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Inizializza il modello actor-critic
    actor_critic = ActorCritic(input_dim=input_dim, num_actions=num_actions).to(device)
    optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)
    
    episode_rewards = []
    episode_final_distances = []
    all_losses = []
    
    for iteration in range(total_iterations):
        # Raccogli un rollout
        storage = collect_trajectories(env, actor_critic, device, rollout_length)
        returns, advantages = compute_ret_adv(storage, gamma)
        
        # Converti i dati raccolti in tensori
        states = torch.tensor(np.array([step['state'] for step in storage]), dtype=torch.float32).to(device)
        actions = torch.tensor([step['action'] for step in storage], dtype=torch.long).to(device)
        old_log_probs = torch.tensor([step['log_prob'] for step in storage], dtype=torch.float32).to(device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32).to(device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(device)
        # Normalizza gli advantage per una migliore stabilità numerica
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        loss = update_policy(actor_critic, optimizer, states, actions, old_log_probs, returns_tensor, advantages_tensor,
                             clip_param=clip_param, ppo_epochs=ppo_epochs, mini_batch_size=mini_batch_size)
        all_losses.append(loss)
        
        # Per il monitoraggio, esegui un episodio di test in modalità greedy
        state, _ = env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            logits, value = actor_critic(state_tensor)
            action = torch.argmax(logits, dim=1).item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
            done = terminated or truncated
        final_distance = np.linalg.norm(state[:3] - env.unwrapped.target)
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
input_dim = 8
num_actions = 9
total_iterations = 500
rollout_length = 100
ppo_epochs = 10
mini_batch_size = 64
clip_param = 0.2

actor_critic, env, rewards_log, distances_log = train_and_plot(input_dim=input_dim,
                                                               num_actions=num_actions,
                                                               total_iterations=total_iterations,
                                                               rollout_length=rollout_length,
                                                               ppo_epochs=ppo_epochs,
                                                               mini_batch_size=mini_batch_size,
                                                               clip_param=clip_param)