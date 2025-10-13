"""
F-Mean Hybrid Layer Implementation

This module implements a hybrid neuron layer that combines standard linear
aggregation with learnable F-mean (power mean) aggregation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HybridFMeanLayer(nn.Module):
    """
    Hybrid layer combining standard linear aggregation with F-mean aggregation.
    
    The layer blends two computational paths:
    1. Linear path: Standard weighted sum (baseline)
    2. F-Mean path: Power-weighted aggregation with learnable parameter p
    
    Parameters:
        in_features (int): Number of input features
        out_features (int): Number of output features
        
    Learnable Parameters:
        weights: Standard neuron weights [out_features, in_features]
        bias: Standard neuron bias [out_features]
        log_p: Log-space power parameter for F-mean [out_features]
        alpha: Blending weight between paths [out_features]
    
    Example:
        >>> layer = HybridFMeanLayer(128, 256)
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
        
        # Novel parameters
        self.log_p = nn.Parameter(torch.zeros(out_features))  # p = exp(log_p), initialized to p=1
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
        
        # Path 2: F-Mean aggregation
        # Compute weighted inputs: z[b, o, i] = x[b, i] * weights[o, i]
        z = x.unsqueeze(1) * self.weights  # Shape: [batch, out_features, in_features]
        
        # Apply softplus for numerical stability (ensures positive values)
        z_pos = F.softplus(z)
        
        # Get power parameter p from log-space
        p = torch.exp(self.log_p)
        p_un = p.view(1, -1, 1)  # Reshape for broadcasting: [1, out_features, 1]
        
        # Compute power transformation: z^p
        z_pos_p = torch.pow(z_pos + self.epsilon, p_un)
        sum_z_pos_p = torch.sum(z_pos_p, dim=2, keepdim=True)
        
        # Compute aggregation weights: w_i = z_i^p / sum(z_j^p)
        agg_weights = z_pos_p / (sum_z_pos_p + self.epsilon)
        
        # F-Mean output: sum(w_i * z_i)
        fmean_out = torch.sum(agg_weights * z, dim=2)
        
        # Hybrid blending with sigmoid-activated alpha
        alpha_clamped = torch.sigmoid(self.alpha)
        output = alpha_clamped * fmean_out + (1 - alpha_clamped) * linear_out
        
        return output
    
    def extra_repr(self) -> str:
        """String representation for printing the layer."""
        return f'in_features={self.in_features}, out_features={self.out_features}'