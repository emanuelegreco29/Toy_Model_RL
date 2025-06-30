import torch
import torch.nn as nn
import torch.distributions as dist
import gymnasium as gym
from .mlp import MLPLayer

class ACTLayer(nn.Module):
    
    def __init__(self, act_space, input_dim, hidden_size, activation_id):
        super().__init__()
        
        # Use custom MLP layer like in original code
        self._mlp_actlayer = False
        if len(hidden_size) > 0:
            self._mlp_actlayer = True
            self.mlp = MLPLayer(input_dim, hidden_size, activation_id)
            input_dim = self.mlp.output_size
            
        # Action head based on space type
        if isinstance(act_space, gym.spaces.Discrete):
            self.action_head = self._init_layer(nn.Linear(input_dim, act_space.n))
            self.action_type = 'discrete'
            
        elif isinstance(act_space, gym.spaces.Box):
            # Mean and log_std for continuous actions
            self.action_mean = self._init_layer(nn.Linear(input_dim, act_space.shape[0]))
            self.action_log_std = nn.Parameter(
                torch.zeros(act_space.shape[0])
            )
            self.action_type = 'continuous'
            
        elif isinstance(act_space, gym.spaces.MultiBinary):
            self.action_head = self._init_layer(nn.Linear(input_dim, act_space.shape[0]))
            self.action_type = 'multibinary'
            
        else:
            raise NotImplementedError(f"Space {type(act_space)} not supported")
    
    def _init_layer(self, layer):
        """Initialize layer weights with proper gain"""
        nn.init.orthogonal_(layer.weight, gain=1.0)
        nn.init.constant_(layer.bias, 0)
        return layer
    
    def forward(self, x, deterministic=False):
        if self._mlp_actlayer:
            x = self.mlp(x)
            
        if self.action_type == 'discrete':
            logits = self.action_head(x)
            action_dist = dist.Categorical(logits=logits)
            
        elif self.action_type == 'continuous':
            mean = self.action_mean(x)
            std = torch.exp(self.action_log_std.expand_as(mean))
            action_dist = dist.Normal(mean, std)
            
        elif self.action_type == 'multibinary':
            logits = self.action_head(x)
            action_dist = dist.Bernoulli(logits=logits)
            
        # Sample or take mode
        if deterministic:
            if self.action_type == 'continuous':
                action = action_dist.mean
            else:
                action = action_dist.mode
        else:
            action = action_dist.sample()
            
        # Calculate log probabilities
        log_prob = action_dist.log_prob(action)
        if len(log_prob.shape) > 1:  # Multi-dimensional actions
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def evaluate_actions(self, x, actions):
        if self._mlp_actlayer:
            x = self.mlp(x)
            
        if self.action_type == 'discrete':
            logits = self.action_head(x)
            action_dist = dist.Categorical(logits=logits)
            
        elif self.action_type == 'continuous':
            mean = self.action_mean(x)
            std = torch.exp(self.action_log_std.expand_as(mean))
            action_dist = dist.Normal(mean, std)
            
        elif self.action_type == 'multibinary':
            logits = self.action_head(x)
            action_dist = dist.Bernoulli(logits=logits)
        
        log_prob = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        
        if len(log_prob.shape) > 1:
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            entropy = entropy.sum(dim=-1, keepdim=True)
            
        return log_prob, entropy