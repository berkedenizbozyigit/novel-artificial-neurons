"""
Novel Hybrid Neuron Layers

This package contains implementations of novel artificial neuron layers
that explore alternative aggregation functions beyond traditional weighted sums.
"""

from .hybrid_fmean import HybridFMeanLayer
from .hybrid_gaussian import HybridGaussianLayer
from .hybrid_three_way import HybridGaussianFMeanLayer

__all__ = [
    'HybridFMeanLayer',
    'HybridGaussianLayer',
    'HybridGaussianFMeanLayer',
]