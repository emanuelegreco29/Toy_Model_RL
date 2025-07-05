import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np

class DiagGaussianDistribution:
    """
    Custom implementation of Diagonal Gaussian Distribution for continuous actions.
    """
    
    def __init__(self, action_dim: int):
        self.action_dim = action_dim
        self.distribution = None
        
    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0):
        """
        Create the layers and parameters that represent the distribution
        
        Args:
            latent_dim: Dimension of the last layer of the policy (before action layer)
            log_std_init: Initial value for the log standard deviation
            
        Returns:
            action_net: Linear layer for mean
            log_std: Parameter for log standard deviation
        """
        # Mean network
        action_net = nn.Linear(latent_dim, self.action_dim)
        # Initialize with small weights
        nn.init.orthogonal_(action_net.weight, gain=0.01)
        nn.init.constant_(action_net.bias, 0.0)
        
        # Log std parameter (independent of input)
        log_std = nn.Parameter(torch.ones(self.action_dim) * log_std_init)
        
        return action_net, log_std
    
    def proba_distribution(self, mean_actions: torch.Tensor, log_std: torch.Tensor):
        """
        Create the distribution given mean and log_std
        
        Args:
            mean_actions: Mean of the actions
            log_std: Log standard deviation of the actions
            
        Returns:
            self for method chaining
        """
        # Clamp log_std to avoid numerical issues
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        # Create standard deviation from log_std
        std = torch.exp(log_std)
        
        # Expand std to match mean_actions batch size
        if mean_actions.dim() == 2:  # (batch_size, action_dim)
            std = std.expand_as(mean_actions)
        
        # Create the distribution
        self.distribution = dist.Normal(mean_actions, std)
        return self
    
    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Get the log probability of actions
        
        Args:
            actions: Actions to evaluate
            
        Returns:
            Log probabilities summed over action dimensions
        """
        if self.distribution is None:
            raise ValueError("Distribution not initialized. Call proba_distribution first.")
        
        # Get log prob for each action dimension and sum
        log_prob = self.distribution.log_prob(actions)
        
        # Sum over action dimensions but keep batch dimension
        return log_prob.sum(dim=-1)
    
    def entropy(self) -> torch.Tensor:
        """
        Get the entropy of the distribution
        
        Returns:
            Entropy summed over action dimensions
        """
        if self.distribution is None:
            raise ValueError("Distribution not initialized. Call proba_distribution first.")
        
        # Sum over action dimensions but keep batch dimension
        return self.distribution.entropy().sum(dim=-1)
    
    def sample(self) -> torch.Tensor:
        """
        Sample actions from the distribution
        
        Returns:
            Sampled actions
        """
        if self.distribution is None:
            raise ValueError("Distribution not initialized. Call proba_distribution first.")
            
        return self.distribution.sample()
    
    def mode(self) -> torch.Tensor:
        """
        Get the mode (mean) of the distribution
        
        Returns:
            Mode of the distribution (mean for Gaussian)
        """
        if self.distribution is None:
            raise ValueError("Distribution not initialized. Call proba_distribution first.")
            
        return self.distribution.mean
    
    def rsample(self) -> torch.Tensor:
        """
        Sample actions from the distribution using reparameterization trick
        
        Returns:
            Sampled actions with gradients
        """
        if self.distribution is None:
            raise ValueError("Distribution not initialized. Call proba_distribution first.")
            
        return self.distribution.rsample()
    
    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        """
        Get actions from the distribution
        
        Args:
            deterministic: Whether to use deterministic actions (mode) or sample
            
        Returns:
            Actions
        """
        if deterministic:
            return self.mode()
        else:
            return self.sample()
    
    def actions_from_params(self, mean_actions: torch.Tensor, log_std: torch.Tensor, 
                          deterministic: bool = False) -> torch.Tensor:
        """
        Sample actions from distribution parameters
        
        Args:
            mean_actions: Mean of actions
            log_std: Log standard deviation
            deterministic: Whether to use deterministic actions
            
        Returns:
            Actions
        """
        self.proba_distribution(mean_actions, log_std)
        return self.get_actions(deterministic)
    
    def log_prob_from_params(self, mean_actions: torch.Tensor, log_std: torch.Tensor, 
                           actions: torch.Tensor) -> torch.Tensor:
        """
        Get log probability from distribution parameters
        
        Args:
            mean_actions: Mean of actions
            log_std: Log standard deviation
            actions: Actions to evaluate
            
        Returns:
            Log probabilities
        """
        self.proba_distribution(mean_actions, log_std)
        return self.log_prob(actions)
    
    def entropy_from_params(self, mean_actions: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        """
        Get entropy from distribution parameters
        
        Args:
            mean_actions: Mean of actions
            log_std: Log standard deviation
            
        Returns:
            Entropy
        """
        self.proba_distribution(mean_actions, log_std)
        return self.entropy()