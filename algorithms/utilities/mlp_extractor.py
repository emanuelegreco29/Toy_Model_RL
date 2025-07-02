import torch
import torch.nn as nn
from typing import Dict, List

class CustomMlpExtractor(nn.Module):
    def __init__(self, 
                 feature_dim: int,
                 net_arch: Dict[str, List[int]],
                 activation_fn: nn.Module = nn.Tanh,
                 device: str = "cpu"):
        """
        Args:
            feature_dim: Dimension of the feature input
            net_arch: Dictionary with 'pi' and 'vf' keys containing layer sizes
            activation_fn: Activation function to use
            device: Device to run on
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.device = device
        
        # Policy network architecture
        policy_net_arch = net_arch.get("pi", [])
        value_net_arch = net_arch.get("vf", [])
        
        # Build policy network
        if len(policy_net_arch) > 0:
            policy_layers = []
            input_dim = feature_dim
            
            for hidden_size in policy_net_arch:
                policy_layers.append(nn.Linear(input_dim, hidden_size))
                policy_layers.append(activation_fn())
                input_dim = hidden_size
            
            self.policy_net = nn.Sequential(*policy_layers)
            self.latent_dim_pi = input_dim
        else:
            self.policy_net = nn.Identity()
            self.latent_dim_pi = feature_dim
        
        # Build value network
        if len(value_net_arch) > 0:
            value_layers = []
            input_dim = feature_dim
            
            for hidden_size in value_net_arch:
                value_layers.append(nn.Linear(input_dim, hidden_size))
                value_layers.append(activation_fn())
                input_dim = hidden_size
            
            self.value_net = nn.Sequential(*value_layers)
            self.latent_dim_vf = input_dim
        else:
            self.value_net = nn.Identity()
            self.latent_dim_vf = feature_dim
    
    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both policy and value networks
        
        Args:
            features: Input features
            
        Returns:
            Tuple of (policy_latent, value_latent)
        """
        policy_latent = self.policy_net(features)
        value_latent = self.value_net(features)
        
        return policy_latent, value_latent
    
    def apply(self, fn):
        """Apply function to all parameters (for initialization)"""
        super().apply(fn)


def create_mlp_layers(input_dim: int, 
                     layer_sizes: List[int], 
                     activation_fn: nn.Module = nn.Tanh) -> nn.Sequential:
    """
    Helper function to create MLP layers
    
    Args:
        input_dim: Input dimension
        layer_sizes: List of hidden layer sizes
        activation_fn: Activation function
        
    Returns:
        Sequential module with the MLP layers
    """
    if not layer_sizes:
        return nn.Identity()
    
    layers = []
    current_dim = input_dim
    
    for size in layer_sizes:
        layers.append(nn.Linear(current_dim, size))
        layers.append(activation_fn())
        current_dim = size
    
    return nn.Sequential(*layers)