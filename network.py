import torch
import torch.nn as nn

def huber_loss(q_values, target_q_values):
    """
    Calcola la loss per la rete Q-Network utilizzando la funzione di perdita Smooth L1 (Huber).
    La loss  viene calcolata tra i Q-values predetti e quelli target.

    Parameters:
        q_values (torch.Tensor): Tensor dei Q-values predetti
        target_q_values (torch.Tensor): Tensor dei Q-values target

    Returns:
        torch.Tensor: Tensor della loss
    """
    loss_fn = nn.SmoothL1Loss()
    return loss_fn(q_values, target_q_values)

class QNetwork(nn.Module):
    def __init__(self, input_dim=5, output_dim=6):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=100000):
        from collections import deque
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        import random
        import numpy as np
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return {
            'obs': torch.tensor(states, dtype=torch.float32),
            'actions': torch.tensor(actions, dtype=torch.long).unsqueeze(1),
            'reward': torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
            'next_state': torch.tensor(next_states, dtype=torch.float32),
            'done': torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        }

    def __len__(self):
        return len(self.buffer)