"""
Neural network architecture for the diffusion model.
"""

import torch
import torch.nn as nn


class DiffusionModel(nn.Module):
    """
    Simple diffusion model architecture.
    """
    
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=3):
        super(DiffusionModel, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        """Forward pass through the model."""
        h = self.encoder(x)
        out = self.decoder(h)
        return out
