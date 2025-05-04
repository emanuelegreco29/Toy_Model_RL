import torch
import torch.nn as nn
from torch.distributions import Normal, Independent

class PPO(nn.Module):
    """
    Actor-Critic network for continuous PPO with adaptive std prediction.
    Actor: two-layer MLP (256,256) with Tanh
    Critic: two-layer MLP (128,128) with LayerNorm + LeakyReLU
    """
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        
        # Actor network
        self.actor_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
        )
        self.mu_layer = nn.Linear(256, act_dim)
        self.log_std_layer = nn.Linear(256, act_dim)

        # Critic network
        self.critic_net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
        )
        self.vf_layer = nn.Linear(128, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Critic head small gain, log_std small gain, else default
                if m is self.vf_layer:
                    gain = 0.01
                elif m is self.log_std_layer:
                    gain = 0.1
                else:
                    gain = 1.0
                nn.init.orthogonal_(m.weight, gain)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, obs: torch.Tensor):
        # Actor forward
        x_a = self.actor_net(obs)
        mu = self.mu_layer(x_a)
        log_std = self.log_std_layer(x_a)
        std = torch.exp(log_std)
        dist = Independent(Normal(mu, std), reinterpreted_batch_ndims=1)

        # Critic forward
        x_c = self.critic_net(obs)
        value = self.vf_layer(x_c).squeeze(-1)
        return dist, value

    def get_action(self, obs: torch.Tensor):
        """
        Sample action, compute log probability and state value.
        """
        dist, value = self(obs)
        action = dist.rsample()
        logp = dist.log_prob(action)
        return action, logp, value

    def get_value(self, obs: torch.Tensor):
        """
        Get state value for given observation.
        """
        _, value = self(obs)
        return value