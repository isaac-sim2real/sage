import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

from .multi_res_branch_net import MultiResolutionBranchNet

class TrunkNet(nn.Module):
    """Trunk network of DeepONet architecture"""
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ELU(),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DeepONetActorCritic(nn.Module):
    """DeepONet-based Actor-Critic network for PPO with multi-resolution branch network"""
    def __init__(
        self,
        branch_input_dims: List[int],  # List of input dimensions for different resolutions
        trunk_input_dim: int,
        action_dim: int,
        branch_hidden_dim: int,
        trunk_hidden_dims: list[int],
        critic_hidden_dims: list[int],
        activation: str = "elu"
    ):
        super().__init__()
        
        # Initialize networks
        self.branch_net = MultiResolutionBranchNet(
            input_dims=branch_input_dims,
            hidden_dim=branch_hidden_dim,
            output_dim=128,
            activation=activation
        )
        self.trunk_net = TrunkNet(trunk_input_dim, trunk_hidden_dims, 128)
        
        # Actor output layer
        self.actor_output = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Tanh()  # Bound actions to [-1, 1]
        )
        
        # Critic network
        critic_layers = []
        prev_dim = trunk_input_dim
        for hidden_dim in critic_hidden_dims:
            critic_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ELU(),
            ])
            prev_dim = hidden_dim
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.critic = nn.Sequential(*critic_layers)
        
        # Action distribution parameters
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, branch_inputs: List[torch.Tensor], trunk_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network
        
        Args:
            branch_inputs: List of inputs for different resolutions of branch network
            trunk_input: Input for trunk network
            
        Returns:
            Tuple of (actions, value)
        """
        # Process through DeepONet
        branch_out = self.branch_net(branch_inputs)
        trunk_out = self.trunk_net(trunk_input)
        combined = branch_out * trunk_out
        
        # Get actions and value
        actions = self.actor_output(combined)
        value = self.critic(trunk_input)
        
        return actions, value
    
    def act(self, branch_inputs: List[torch.Tensor], trunk_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get actions and their log probabilities for training
        
        Args:
            branch_inputs: List of inputs for different resolutions of branch network
            trunk_input: Input for trunk network
            
        Returns:
            Tuple of (actions, log_probs, entropy)
        """
        actions, _ = self.forward(branch_inputs, trunk_input)
        
        # Create normal distribution
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(actions, std)
        
        # Sample actions and compute log probabilities
        actions = dist.sample()
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return actions, log_probs, entropy
    
    def evaluate(self, branch_inputs: List[torch.Tensor], trunk_input: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions and compute value estimates
        
        Args:
            branch_inputs: List of inputs for different resolutions of branch network
            trunk_input: Input for trunk network
            actions: Actions to evaluate
            
        Returns:
            Tuple of (value, log_probs, entropy)
        """
        # Get value estimate
        value = self.critic(trunk_input)
        
        # Process through DeepONet
        branch_out = self.branch_net(branch_inputs)
        trunk_out = self.trunk_net(trunk_input)
        combined = branch_out * trunk_out
        
        # Get mean actions
        mean_actions = self.actor_output(combined)
        
        # Create normal distribution
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean_actions, std)
        
        # Compute log probabilities and entropy
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return value, log_probs, entropy