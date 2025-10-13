"""
Neural Network Models with Hybrid Layers

This package contains MLP and CNN architectures that utilize novel hybrid
neuron layers for classification tasks.
"""

from .mlp import HybridMLP, BaselineMLP
from .cnn import HybridCNN, BaselineCNN

__all__ = [
    'HybridMLP',
    'BaselineMLP',
    'HybridCNN',
    'BaselineCNN',
]