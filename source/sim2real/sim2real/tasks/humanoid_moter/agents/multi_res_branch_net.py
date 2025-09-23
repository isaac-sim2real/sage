import torch
import torch.nn as nn
from typing import List, Tuple

class MultiResolutionBranchNet(nn.Module):
    """Multi-resolution branch network for DeepONet architecture"""
    def __init__(
        self,
        input_dims: List[int],  # List of input dimensions for different resolutions
        hidden_dim: int,
        output_dim: int,
        activation: str = "elu"
    ):
        super().__init__()
        
        # Set activation function
        self.activation = nn.ELU() if activation == "elu" else nn.ReLU()
        
        # Create branches for different resolutions
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                self.activation,
                nn.Linear(hidden_dim, hidden_dim),
                self.activation
            ) for input_dim in input_dims
        ])
        
        # Fusion layer to combine multi-resolution features
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * len(input_dims), output_dim),
            self.activation
        )
    
    def forward(self, x_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the multi-resolution branch network
        
        Args:
            x_list: List of inputs for different resolutions
            
        Returns:
            Combined multi-resolution features
        """
        # Process each resolution through its branch
        branch_outputs = [branch(x) for branch, x in zip(self.branches, x_list)]
        
        # Concatenate all branch outputs
        combined = torch.cat(branch_outputs, dim=-1)
        
        # Fuse the features
        return self.fusion(combined)