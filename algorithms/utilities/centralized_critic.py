import torch.nn as nn

# Critic che prende in ingresso lo stato globale
class CentralizedCritic(nn.Module):
    def __init__(self, global_state_dim, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, global_state):
        # restituisce V(s) scalare
        return self.net(global_state).squeeze(-1)