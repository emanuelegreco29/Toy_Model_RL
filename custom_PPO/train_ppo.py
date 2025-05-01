import os
import datetime
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from environment import PointMassEnv
from ppo_network import ActorCritic

def collect_trajectories(env, actor_critic, device, rollout_length, gamma):
    """
    Esegue un rollout per `rollout_length` passi e ritorna arrays:
    states, actions, old_log_probs, returns, advantages
    """
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device)

    states, actions, old_log_probs, rewards, dones, values = [], [], [], [], [], []
    action_dim = env.action_space.shape[0]
    low  = torch.tensor(env.action_space.low, dtype=torch.float32, device=device).unsqueeze(0)
    high = torch.tensor(env.action_space.high, dtype=torch.float32, device=device).unsqueeze(0)

    for _ in range(rollout_length):
        mean, std, value = actor_critic(obs.unsqueeze(0))
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action_clamped = torch.clamp(action, min=low, max=high)
        action_np = action_clamped.squeeze(0).cpu().numpy()
        log_prob = dist.log_prob(action).sum(dim=-1)

        next_obs, reward, terminated, truncated, _ = env.step(action_np)
        done = terminated or truncated

        states.append(obs.cpu().numpy())
        actions.append(action.cpu().numpy())
        old_log_probs.append(log_prob.cpu().detach().numpy())
        rewards.append(reward)
        dones.append(done)
        values.append(value.cpu().detach().numpy().squeeze())

        obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
        if done:
            obs, _ = env.reset()
            obs = torch.tensor(obs, dtype=torch.float32, device=device)

    # bootstrap dal valore dell'ultima osservazione
    _, _, last_value = actor_critic(obs.unsqueeze(0))
    last_value = last_value.cpu().detach().numpy().squeeze()

    # calcola returns e advantage
    returns = []
    G = last_value
    for r, done in zip(reversed(rewards), reversed(dones)):
        if done:
            G = 0.0
        G = r + gamma * G
        returns.insert(0, G)
    returns = np.array(returns)
    values = np.array(values)
    advantages = returns - values

    return (
        np.array(states),
        np.array(actions),
        np.array(old_log_probs),
        returns,
        advantages
    )

def train_and_plot(input_dim, num_actions, total_iterations, rollout_length, ppo_epochs, mini_batch_size, learning_rate, entropy_coeff, clip_param):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = PointMassEnv()
    actor_critic = ActorCritic(input_dim=input_dim, num_actions=num_actions).to(device)
    optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)

    episode_rewards = []
    episode_final_distances = []
    all_losses = []

    gamma = 0.99

    for iteration in range(total_iterations):
        # Raccolta traiettorie
        states, actions, old_log_probs, returns, advantages = collect_trajectories(
            env, actor_critic, device, rollout_length, gamma
        )
        # normalizza vantaggi
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset = TensorDataset(
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32),
            torch.tensor(old_log_probs, dtype=torch.float32),
            torch.tensor(returns, dtype=torch.float32),
            torch.tensor(advantages, dtype=torch.float32)
        )
        loader = DataLoader(dataset, batch_size=mini_batch_size, shuffle=True)

        epoch_loss = 0.0
        for _ in range(ppo_epochs):
            for b_states, b_actions, b_old_log_probs, b_returns, b_advantages in loader:
                b_states = b_states.to(device)
                b_actions = b_actions.to(device)
                b_old_log_probs = b_old_log_probs.to(device)
                b_returns = b_returns.to(device)
                b_advantages = b_advantages.to(device)

                mean, std, values = actor_critic(b_states)
                dist = torch.distributions.Normal(mean, std)
                new_log_probs = dist.log_prob(b_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(new_log_probs - b_old_log_probs)
                clipped = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * b_advantages
                policy_loss = -torch.min(ratio * b_advantages, clipped).mean()
                value_loss = (b_returns - values.squeeze(-1)).pow(2).mean()
                loss = policy_loss + 0.5 * value_loss - entropy_coeff * entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

        avg_loss = epoch_loss / (ppo_epochs * len(loader))
        all_losses.append(avg_loss)

        # Valutazione episodio per logging
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                mean, _, _ = actor_critic(obs_tensor)
                action_np = mean.cpu().numpy()[0]
                action_clamped = np.clip(action_np,
                                        env.action_space.low,
                                        env.action_space.high)

            next_obs, reward, terminated, truncated, _ = env.step(action_clamped)
            done = terminated or truncated
            episode_reward += reward
            obs = next_obs

        final_distance = np.linalg.norm(obs[:3] - env.target)
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
input_dim = 5
num_actions = 2
total_iterations = 500
rollout_length = 500
ppo_epochs = 10
mini_batch_size = 64
learning_rate = 0.0005
entropy_coeff = 0.02
clip_param = 0.2

train_and_plot(
    input_dim=input_dim,
    num_actions=num_actions,
    total_iterations=total_iterations,
    rollout_length=rollout_length,
    ppo_epochs=ppo_epochs,
    mini_batch_size=mini_batch_size,
    learning_rate=learning_rate,
    entropy_coeff=entropy_coeff,
    clip_param=clip_param
)