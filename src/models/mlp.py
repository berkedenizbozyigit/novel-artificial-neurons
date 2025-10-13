"""
Multi-Layer Perceptron with Hybrid Layers

This module implements MLP architectures using novel hybrid neuron layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridMLP(nn.Module):
    """
    MLP model with projection layer and hybrid neuron layers.
    
    Architecture:
        Input -> Projection Layer -> Hybrid Layer -> Classifier -> Output
    
    The projection layer reduces dimensionality before the hybrid layer to
    prevent memory issues with quadratic complexity operations (especially
    for Gaussian layers).
    
    Parameters:
        hybrid_layer_class: Class of hybrid layer to use (HybridFMeanLayer, etc.)
        input_dim (int): Input feature dimension (e.g., 3072 for flattened CIFAR-10)
        projection_dim (int): Dimension after projection (reduces memory usage)
        hidden_dim (int): Hidden dimension of hybrid layer
        output_dim (int): Number of output classes (e.g., 10 for CIFAR-10)
    
    Example:
        >>> from src.layers import HybridFMeanLayer
        >>> model = HybridMLP(HybridFMeanLayer, 3072, 128, 256, 10)
        >>> x = torch.randn(32, 3072)  # batch of flattened images
        >>> output = model(x)  # shape: [32, 10]
    """
    
    def __init__(
        self,
        hybrid_layer_class,
        input_dim: int,
        projection_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 10
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Projection layer: reduces dimensionality
        self.projection = nn.Linear(input_dim, projection_dim)
        
        # Hybrid layer: novel aggregation function
        self.hybrid_layer = hybrid_layer_class(projection_dim, hidden_dim)
        
        # Classifier: standard linear layer
        self.classifier = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Output logits of shape [batch_size, output_dim]
        """
        # Flatten input if needed (for images)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Project to lower dimension with ReLU activation
        x = F.relu(self.projection(x))
        
        # Pass through hybrid layer with ReLU activation
        x = F.relu(self.hybrid_layer(x))
        
        # Final classification layer (no activation, for use with CrossEntropyLoss)
        x = self.classifier(x)
        
        return x
    
    def get_hybrid_parameters(self):
        """
        Returns novel parameters (p, sigma, alpha) separate from standard weights.
        
        Returns:
            List of novel parameters for differential learning rates
        """
        novel_params = []
        for name, param in self.named_parameters():
            if any(novel_name in name for novel_name in ['p', 'log_p', 'alpha', 'log_sigma', 'alphas']):
                novel_params.append(param)
        return novel_params
    
    def get_standard_parameters(self):
        """
        Returns standard parameters (weights, biases) separate from novel params.
        
        Returns:
            List of standard parameters for differential learning rates
        """
        standard_params = []
        for name, param in self.named_parameters():
            if not any(novel_name in name for novel_name in ['p', 'log_p', 'alpha', 'log_sigma', 'alphas']):
                standard_params.append(param)
        return standard_params


class BaselineMLP(nn.Module):
    """
    Baseline MLP model using only standard linear layers (for comparison).
    
    Architecture:
        Input -> Projection Layer -> Hidden Layer -> Classifier -> Output
    
    Parameters:
        input_dim (int): Input feature dimension
        projection_dim (int): Dimension after projection
        hidden_dim (int): Hidden layer dimension
        output_dim (int): Number of output classes
    
    Example:
        >>> model = BaselineMLP(3072, 128, 256, 10)
        >>> x = torch.randn(32, 3072)
        >>> output = model(x)  # shape: [32, 10]
    """
    
    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 10
    ):
        super().__init__()
        
        self.projection = nn.Linear(input_dim, projection_dim)
        self.hidden = nn.Linear(projection_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through baseline network."""
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        x = F.relu(self.projection(x))
        x = F.relu(self.hidden(x))
        x = self.classifier(x)
        
        return x