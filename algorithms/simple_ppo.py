import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Type, Any
import gymnasium as gym

class SimpleLSTMExtractor(BaseFeaturesExtractor):
    """
    Semplice feature extractor con LSTM per catturare la dinamica temporale
    del target tracking
    """
    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        
        # Dimensioni input
        obs_dim = observation_space.shape[0]
        
        # Separiamo le componenti dell'osservazione
        # [agent_state(6) + target_history(K+1)*3 + target_dynamics(3)]
        self.agent_dim = 6
        self.history_dim = obs_dim - 6 - 3  # tutto tranne agent e dynamics
        self.dynamics_dim = 3
        
        # MLP per stato agente
        self.agent_mlp = nn.Sequential(
            nn.Linear(self.agent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # LSTM per history del target (se presente)
        if self.history_dim > 0:
            # Assumiamo history_dim = (K+1)*3, quindi K+1 timesteps di posizioni 3D
            self.n_history_steps = self.history_dim // 3
            self.history_lstm = nn.LSTM(
                input_size=3,  # posizione 3D ad ogni timestep
                hidden_size=32,
                num_layers=1,
                batch_first=True
            )
        else:
            self.n_history_steps = 0
            self.history_lstm = None
        
        # MLP per target dynamics
        self.dynamics_mlp = nn.Sequential(
            nn.Linear(self.dynamics_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        
        # Fusion layer
        fusion_input_dim = 32  # agent
        if self.history_lstm is not None:
            fusion_input_dim += 32  # LSTM output
        fusion_input_dim += 16  # dynamics
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim)
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        # Separiamo i componenti
        agent_state = observations[:, :self.agent_dim]
        
        if self.history_dim > 0:
            history_flat = observations[:, self.agent_dim:self.agent_dim + self.history_dim]
            # Reshape per LSTM: (batch, sequence_length, features)
            history = history_flat.view(batch_size, self.n_history_steps, 3)
        
        dynamics = observations[:, -self.dynamics_dim:]
        
        # Processa agent state
        agent_features = self.agent_mlp(agent_state)
        
        # Processa history con LSTM
        if self.history_lstm is not None:
            lstm_out, _ = self.history_lstm(history)
            # Prendiamo l'ultimo output
            history_features = lstm_out[:, -1, :]
        else:
            history_features = torch.zeros(batch_size, 32, device=observations.device)
        
        # Processa dynamics
        dynamics_features = self.dynamics_mlp(dynamics)
        
        # Fusione
        if self.history_lstm is not None:
            combined = torch.cat([agent_features, history_features, dynamics_features], dim=1)
        else:
            combined = torch.cat([agent_features, dynamics_features], dim=1)
        
        return self.fusion(combined)

class SimplePolicy(ActorCriticPolicy):
    """
    Policy semplice ma efficace per il target tracking
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, 
                         net_arch=dict(pi=[64, 64], vf=[64, 64]))

def create_simple_ppo_model(env, learning_rate=3e-4):
    """
    Crea un modello PPO semplificato ma efficace
    """
    
    # Policy con feature extractor custom
    policy_kwargs = dict(
        features_extractor_class=SimpleLSTMExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[64, 64], vf=[64, 64]),  # reti più piccole
        activation_fn=nn.ReLU
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=2048,        # Ridotto per training più frequente
        batch_size=64,       # Batch size piccolo
        n_epochs=4,          # Meno epochs per evitare overfitting
        gamma=0.99,          # Discount factor standard
        gae_lambda=0.95,     # GAE parameter
        clip_range=0.2,      # Standard PPO clip
        ent_coef=0.01,       # Entropia ridotta (più exploitation)
        vf_coef=0.5,         # Value function coefficient
        max_grad_norm=0.5,   # Gradient clipping
        policy_kwargs=policy_kwargs,
        verbose=1,
        device="cpu"
    )
    
    return model

class ReplayBuffer:
    """
    Semplice replay buffer per migliorare l'apprendimento
    """
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, obs, action, reward, next_obs, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (obs, action, reward, next_obs, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        import random
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done = map(np.stack, zip(*batch))
        return obs, action, reward, next_obs, done
    
    def __len__(self):
        return len(self.buffer)

class EnhancedEpisodeLogger:
    """
    Logger migliorato con più metriche
    """
    def __init__(self):
        self.rewards = []
        self.distances = []
        self.followed = []
        self.episode_lengths = []
        self.success_rate = []
        
    def __call__(self, locals_, globals_):
        for info in locals_["infos"]:
            ep = info.get("episode")
            if ep:
                r = ep["r"]
                length = ep["l"]
                d = info.get("distance", np.nan)
                f = info.get("followed", np.nan)
                
                self.rewards.append(r)
                self.distances.append(d)
                self.followed.append(f)
                self.episode_lengths.append(length)
                
                # Success rate (seguire per più del 60% dell'episodio)
                success = f > (length * 0.6) if not np.isnan(f) and length > 0 else False
                self.success_rate.append(success)
                
                n = len(self.rewards)
                avg_reward = np.mean(self.rewards[-10:]) if n >= 10 else np.mean(self.rewards)
                success_rate = np.mean(self.success_rate[-10:]) if n >= 10 else np.mean(self.success_rate)
                
                print(f"Episode {n} | Reward: {r:.2f} | Followed: {f} | Avg10: {avg_reward:.2f} | Success: {success_rate:.2%}")
        return True
    
    def plot_training_progress(self):
        """Plotta i risultati del training"""
        eps = np.arange(1, len(self.rewards) + 1)
        
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        
        # Rewards
        axs[0,0].plot(eps, self.rewards, alpha=0.3, color='blue')
        if len(self.rewards) >= 10:
            moving_avg = np.convolve(self.rewards, np.ones(10)/10, mode='valid')
            axs[0,0].plot(eps[9:], moving_avg, color='blue', linewidth=2)
        axs[0,0].set_title('Reward per Episode')
        axs[0,0].set_xlabel('Episode')
        axs[0,0].set_ylabel('Reward')
        
        # Followed steps
        axs[0,1].plot(eps, self.followed, marker='o', markersize=2, color='orange')
        axs[0,1].set_title('Steps Following Target')
        axs[0,1].set_xlabel('Episode')
        axs[0,1].set_ylabel('Steps')
        
        # Success rate
        if len(self.success_rate) >= 10:
            success_moving = np.convolve([int(x) for x in self.success_rate], 
                                       np.ones(10)/10, mode='valid')
            axs[1,0].plot(eps[9:], success_moving, color='green', linewidth=2)
        axs[1,0].set_title('Success Rate (10-episode moving average)')
        axs[1,0].set_xlabel('Episode')
        axs[1,0].set_ylabel('Success Rate')
        axs[1,0].set_ylim(0, 1)
        
        # Episode lengths
        axs[1,1].plot(eps, self.episode_lengths, color='red', alpha=0.6)
        axs[1,1].set_title('Episode Length')
        axs[1,1].set_xlabel('Episode')
        axs[1,1].set_ylabel('Steps')
        
        plt.tight_layout()
        plt.show()

# Esempio di utilizzo semplificato
def train_simple_ppo(env_creator, episodes=1000, learning_rate=3e-4):
    """
    Training semplificato ma efficace
    """
    # Crea environment
    env = DummyVecEnv([lambda: Monitor(env_creator())])
    
    # Crea modello
    model = create_simple_ppo_model(env, learning_rate)
    
    # Logger
    callback = EnhancedEpisodeLogger()
    
    # Training
    total_timesteps = 500 * episodes
    print(f"Training for {episodes} episodes ({total_timesteps} timesteps)")
    
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    # Plot risultati
    callback.plot_training_progress()
    
    return model, callback