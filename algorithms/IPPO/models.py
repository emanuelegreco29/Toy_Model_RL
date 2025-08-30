import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden=(128, 128), act=nn.Tanh):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), act()]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.net(x)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(128, 128), log_std_init: float = -0.5):
        super().__init__()
        self.actor = MLP(obs_dim, act_dim, hidden)
        self.critic = MLP(obs_dim, 1, hidden)
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor | None = None):
        mean = self.actor(x)
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(x).squeeze(-1)
        return action, log_prob, entropy, value

    def get_value(self, x: torch.Tensor):
        return self.critic(x).squeeze(-1)
    
class EnhancedActorCritic(ActorCritic):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256, 128), log_std_init: float = -0.5):
        # Call parent's parent (nn.Module) to avoid ActorCritic's __init__
        nn.Module.__init__(self)
        
        # Enhanced architectures with dropout and layer norm
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden[0]),
            nn.LayerNorm(hidden[0]),
            nn.Tanh(),
            #nn.Dropout(0.1),
            nn.Linear(hidden[0], hidden[1]),
            nn.LayerNorm(hidden[1]),
            nn.Tanh(),
            #nn.Dropout(0.1),
            nn.Linear(hidden[1], hidden[2]),
            nn.LayerNorm(hidden[2]),
            nn.Tanh(),
            nn.Linear(hidden[2], act_dim)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden[0]),
            nn.LayerNorm(hidden[0]),
            nn.Tanh(),
            #nn.Dropout(0.1),
            nn.Linear(hidden[0], hidden[1]),
            nn.LayerNorm(hidden[1]),
            nn.Tanh(),
            #nn.Dropout(0.1),
            nn.Linear(hidden[1], hidden[2]),
            nn.LayerNorm(hidden[2]),
            nn.Tanh(),
            nn.Linear(hidden[2], 1)
        )
        
        # Initialize weights
        for module in [self.actor, self.critic]:
            for m in module:
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                    nn.init.constant_(m.bias, 0.0)
        
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor | None = None):
        mean = self.actor(x)
        std = self.log_std.exp().expand_as(mean)
        std = torch.clamp(std, min=0.01, max=2.0)  # Clamp std for stability
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(x).squeeze(-1)
        return action, log_prob, entropy, value

    def get_value(self, x: torch.Tensor):
        return self.critic(x).squeeze(-1)