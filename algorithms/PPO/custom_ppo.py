import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

from ..utilities.mlp_extractor import CustomMlpExtractor
from ..utilities.distributions import DiagGaussianDistribution

class CustomActorCritic(nn.Module):
    def __init__(self, 
                 obs_dim: int, 
                 act_dim: int, 
                 net_arch: Tuple[int, ...] = (512, 512, 512), 
                 log_std_init: float = -2.0,
                 device: str = "cpu"):
        super().__init__()
        
        # Fix device handling
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # 1) Feature extractor (Flatten Layer)
        self.features_extractor = nn.Flatten()
        self.features_dim = obs_dim
        
        # 2) Custom MLP extractor: separa policy e value networks
        self.mlp_extractor = CustomMlpExtractor(
            feature_dim=self.features_dim,
            net_arch={"pi": list(net_arch), "vf": list(net_arch)},
            activation_fn=nn.Tanh,
            device=str(self.device),
        )
        
        # 3) Action dimension per la distribuzione
        self.action_dim = act_dim
        
        # 4) Custom distribution per azioni continue
        self.action_dist = DiagGaussianDistribution(act_dim)
        
        # 5) Policy head: action distribution parameters
        self.action_net, self.log_std = self.action_dist.proba_distribution_net(
            latent_dim=self.mlp_extractor.latent_dim_pi,
            log_std_init=log_std_init,
        )
        
        # 6) Value head
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        
        # 7) Inizializzazione ortogonale dei pesi
        self._init_weights()
        
        # Move to device
        self.to(self.device)
    
    def _init_weights(self):
        """Inizializzazione ortogonale dei pesi"""
        
        # MLP extractor - gain sqrt(2) per hidden layers
        def init_mlp_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
        self.mlp_extractor.apply(init_mlp_weights)
        
        # Action net (policy head) - gain piccolo per stabilità
        nn.init.orthogonal_(self.action_net.weight, gain=0.01)
        nn.init.constant_(self.action_net.bias, 0.0)
        
        # Value net - gain standard
        nn.init.orthogonal_(self.value_net.weight, gain=1.0)
        nn.init.constant_(self.value_net.bias, 0.0)
    
    def _get_tensors_on_device(self, *tensors):
        """Ensure tensors are on the correct device"""
        return [t.to(self.device) if isinstance(t, torch.Tensor) else t for t in tensors]
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass che restituisce mean, log_std e values
        
        Args:
            obs: (batch_size, obs_dim)
            
        Returns:
            mean_actions: (batch_size, act_dim)
            log_std: (act_dim,) - parametro della rete
            values: (batch_size, 1)
        """
        obs = obs.to(self.device)
        
        # Estrazione features
        features = self.features_extractor(obs)
        
        # Separazione policy e value networks
        latent_pi, latent_vf = self.mlp_extractor(features)
        
        # Policy output (mean delle azioni)
        mean_actions = self.action_net(latent_pi)
        
        # Value output
        values = self.value_net(latent_vf)
        
        return mean_actions, self.log_std, values
    
    def get_distribution(self, obs: torch.Tensor):
        """
        Metodo per ottenere la distribuzione delle azioni
        
        Args:
            obs: Observations
            
        Returns:
            DiagGaussianDistribution: Distribuzione configurata
        """
        mean_actions, log_std, _ = self.forward(obs)
        return self.action_dist.proba_distribution(mean_actions, log_std)
    
    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Metodo per predire solo i valori
        
        Args:
            obs: Observations
            
        Returns:
            values: Predicted values
        """
        _, _, values = self.forward(obs)
        return values
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO training
        
        Args:
            obs: Observations
            actions: Actions to evaluate
            
        Returns:
            values: Value predictions
            log_prob: Log probabilities of actions
            entropy: Entropy of the action distribution
        """
        obs, actions = self._get_tensors_on_device(obs, actions)
        
        mean_actions, log_std, values = self.forward(obs)
        
        # Configura la distribuzione
        distribution = self.action_dist.proba_distribution(mean_actions, log_std)
        
        # Calcola log_prob ed entropy
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        return values, log_prob, entropy
    
    def predict(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict actions and values
        
        Args:
            obs: Observations
            deterministic: Whether to sample deterministically
            
        Returns:
            actions: Predicted actions
            values: Predicted values
        """
        obs = obs.to(self.device)
        
        mean_actions, log_std, values = self.forward(obs)
        
        # Configura la distribuzione
        distribution = self.action_dist.proba_distribution(mean_actions, log_std)
        
        # Sample o mode
        if deterministic:
            actions = distribution.mode()
        else:
            actions = distribution.sample()
        
        return actions, values
    
    def get_action_log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Get log probability of specific actions
        
        Args:
            obs: Observations
            actions: Actions to evaluate
            
        Returns:
            log_prob: Log probabilities
        """
        obs, actions = self._get_tensors_on_device(obs, actions)
        distribution = self.get_distribution(obs)
        return distribution.log_prob(actions)