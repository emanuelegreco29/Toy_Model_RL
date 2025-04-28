import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, input_dim=11, num_actions=3):
        super(ActorCritic, self).__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        
        # Per la policy continua uso mean e log standard
        self.mean_head = nn.Linear(32, num_actions)
        self.log_std = nn.Parameter(torch.zeros(num_actions))
        # Stima il valore dello stato
        self.value_head = nn.Linear(32, 1)
        
    def forward(self, x):
        shared = self.shared_net(x)
        mean = self.mean_head(shared)
        std  = torch.exp(self.log_std)
        value = self.value_head(shared)
        return mean, std, value
