"""
Utility Functions for Training and Data Processing

This package contains utilities for data loading, training loops,
and evaluation of hybrid neuron models.
"""

from .data_loaders import (
    AddGaussianNoise,
    get_cifar10_loaders,
    get_dataset_info
)

from .training import (
    train_epoch,
    test,
    get_hybrid_parameters,
    print_hybrid_parameters,
    create_optimizer,
    train_model
)

__all__ = [
    # Data loading
    'AddGaussianNoise',
    'get_cifar10_loaders',
    'get_dataset_info',
    
    # Training
    'train_epoch',
    'test',
    'get_hybrid_parameters',
    'print_hybrid_parameters',
    'create_optimizer',
    'train_model',
]