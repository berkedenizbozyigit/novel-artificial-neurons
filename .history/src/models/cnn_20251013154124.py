"""
Convolutional Neural Network with Hybrid Classifier Head

This module implements CNN architectures using novel hybrid neurons in the
classifier head while maintaining standard convolutional feature extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridCNN(nn.Module):
    """
    CNN model with standard convolutional base and hybrid classifier head.
    
    Architecture:
        Input (32x32x3) 
        -> Conv Base (feature extraction)
        -> Flatten (8192 features)
        -> Projection (reduces to 256)
        -> Hybrid Classifier Head
        -> Output (10 classes)
    
    The hybrid classifier head uses novel aggregation functions while the
    convolutional base performs standard spatial feature extraction.
    
    Parameters:
        hybrid_layer_class: Class of hybrid layer to use (HybridFMeanLayer, etc.)
        projection_dim (int): Dimension after projection (default: 256)
        hidden_dim (int): Hidden dimension in classifier (default: 256)
        output_dim (int): Number of output classes (default: 10)
    
    Example:
        >>> from src.layers import HybridFMeanLayer
        >>> model = HybridCNN(HybridFMeanLayer, projection_dim=256, hidden_dim=256)
        >>> x = torch.randn(32, 3, 32, 32)  # CIFAR-10 batch
        >>> output = model(x)  # shape: [32, 10]
    """
    
    def __init__(
        self,
        hybrid_layer_class,
        projection_dim: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 10
    ):
        super().__init__()
        
        # Standard convolutional feature extractor
        # Input: 3x32x32 -> Output: 128x8x8
        self.conv_base = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64x16x16
            
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128x8x8
        )
        
        # Calculate flattened size: 128 * 8 * 8 = 8192 for 32x32 input
        self.conv_output_size = 128 * 8 * 8
        
        # Projection layer for memory efficiency
        self.projection = nn.Sequential(
            nn.Linear(self.conv_output_size, projection_dim),
            nn.ReLU()
        )
        
        # Classifier with hybrid neuron
        self.classifier = nn.Sequential(
            hybrid_layer_class(projection_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, 3, 32, 32]
            
        Returns:
            Output logits of shape [batch_size, output_dim]
        """
        # Feature extraction with convolutional base
        x = self.conv_base(x)  # [batch, 128, 8, 8]
        
        # Flatten spatial dimensions
        x = x.view(x.size(0), -1)  # [batch, 8192]
        
        # Project to lower dimension
        x = self.projection(x)  # [batch, projection_dim]
        
        # Classify with hybrid layer
        x = self.classifier(x)  # [batch, output_dim]
        
        return x
    
    def get_hybrid_parameters(self):
        """
        Returns novel parameters (p, sigma, alpha) for differential learning rates.
        
        Returns:
            List of novel parameters
        """
        novel_params = []
        for name, param in self.named_parameters():
            if any(novel_name in name for novel_name in ['p', 'log_p', 'alpha', 'log_sigma', 'alphas']):
                novel_params.append(param)
        return novel_params
    
    def get_standard_parameters(self):
        """
        Returns standard parameters (weights, biases) for differential learning rates.
        
        Returns:
            List of standard parameters
        """
        standard_params = []
        for name, param in self.named_parameters():
            if not any(novel_name in name for novel_name in ['p', 'log_p', 'alpha', 'log_sigma', 'alphas']):
                standard_params.append(param)
        return standard_params


class BaselineCNN(nn.Module):
    """
    Baseline CNN model using only standard linear layers (for comparison).
    
    Architecture:
        Input -> Conv Base -> Projection -> Linear Classifier -> Output
    
    Parameters:
        projection_dim (int): Dimension after projection
        hidden_dim (int): Hidden dimension in classifier
        output_dim (int): Number of output classes
    
    Example:
        >>> model = BaselineCNN(projection_dim=256, hidden_dim=256)
        >>> x = torch.randn(32, 3, 32, 32)
        >>> output = model(x)  # shape: [32, 10]
    """
    
    def __init__(
        self,
        projection_dim: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 10
    ):
        super().__init__()
        
        # Same convolutional base as hybrid version
        self.conv_base = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.conv_output_size = 128 * 8 * 8
        
        self.projection = nn.Sequential(
            nn.Linear(self.conv_output_size, projection_dim),
            nn.ReLU()
        )
        
        # Standard linear classifier
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through baseline CNN."""
        x = self.conv_base(x)
        x = x.view(x.size(0), -1)
        x = self.projection(x)
        x = self.classifier(x)
        return x