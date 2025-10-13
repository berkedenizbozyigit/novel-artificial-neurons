"""
Gaussian Support Hybrid Layer Implementation

This module implements a hybrid neuron layer that combines standard linear
aggregation with Gaussian affinity-based aggregation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HybridGaussianLayer(nn.Module):
    """
    Hybrid layer combining standard linear aggregation with Gaussian support aggregation.
    
    The layer blends two computational paths:
    1. Linear path: Standard weighted sum (baseline)
    2. Gaussian path: Pairwise affinity-based aggregation with learnable sigma
    
    Parameters:
        in_features (int): Number of input features
        out_features (int): Number of output features
        
    Learnable Parameters:
        weights: Standard neuron weights [out_features, in_features]
        bias: Standard neuron bias [out_features]
        log_sigma: Log-space Gaussian width parameter [out_features]
        alpha: Blending weight between paths [out_features]
    
    Example:
        >>> layer = HybridGaussianLayer(128, 256)
        >>> x = torch.randn(32, 128)  # batch_size=32
        >>> output = layer(x)  # shape: [32, 256]
    """
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.epsilon = 1e-8
        
        # Standard neural network weights and bias
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Gaussian-specific parameters
        self.log_sigma = nn.Parameter(torch.zeros(out_features))  # sigma = exp(log_sigma), initialized to sigma=1
        self.alpha = nn.Parameter(torch.zeros(out_features))  # Initialized to 0.5 after sigmoid
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid layer.
        
        Args:
            x: Input tensor of shape [batch_size, in_features]
            
        Returns:
            Output tensor of shape [batch_size, out_features]
        """
        # Path 1: Standard linear aggregation
        linear_out = F.linear(x, self.weights, self.bias)
        
        # Path 2: Gaussian Support aggregation
        # Compute weighted inputs: z[b, o, i] = x[b, i] * weights[o, i]
        z = x.unsqueeze(1) * self.weights  # Shape: [batch, out_features, in_features]
        
        # Compute pairwise squared distances
        # dist_sq[b, o, i, j] = ||z[b,o,i] - z[b,o,j]||^2
        dist_sq = (z.unsqueeze(3) - z.unsqueeze(2)) ** 2
        
        # Apply Gaussian kernel with learnable sigma
        sigma_sq = (2 * torch.exp(self.log_sigma) ** 2).view(1, self.out_features, 1, 1)
        affinity_matrix = torch.exp(-dist_sq / (sigma_sq + self.epsilon))
        
        # Sum affinities across all inputs to get support scores
        support_scores = torch.sum(affinity_matrix, dim=3)
        
        # Normalize to get aggregation weights (L1 normalization)
        agg_weights = F.normalize(support_scores, p=1, dim=2)
        
        # Gaussian output: weighted sum using affinity-based weights
        gaussian_out = torch.sum(agg_weights * z, dim=2)
        
        # Hybrid blending with sigmoid-activated alpha
        alpha_clamped = torch.sigmoid(self.alpha)
        output = alpha_clamped * gaussian_out + (1 - alpha_clamped) * linear_out
        
        return output
    
    def extra_repr(self) -> str:
        """String representation for printing the layer."""
        return f'in_features={self.in_features}, out_features={self.out_features}'