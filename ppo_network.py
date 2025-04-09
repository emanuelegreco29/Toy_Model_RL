import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, input_dim=8, num_actions=9):
        super(ActorCritic, self).__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        # Fornisce in output i logits (valori grezzi non normalizzati) per le 9 azioni discrete
        self.policy_head = nn.Linear(512, num_actions)
        # Stima il valore dello stato
        self.value_head = nn.Linear(512, 1)
        
    def forward(self, x):
        shared = self.shared_net(x)
        logits = self.policy_head(shared)
        value = self.value_head(shared)
        return logits, value
