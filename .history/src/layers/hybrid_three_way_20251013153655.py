"""
Three-Way Hybrid Layer Implementation

This module implements a hybrid neuron layer that combines three aggregation
methods: linear, F-mean, and Gaussian support, with learnable blending weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HybridGaussianFMeanLayer(nn.Module):
    """
    Three-way hybrid layer combining linear, F-mean, and Gaussian aggregation.
    
    The layer blends three computational paths:
    1. Linear path: Standard weighted sum
    2. F-Mean path: Power-weighted aggregation
    3. Gaussian path: Affinity-based aggregation
    
    Parameters:
        in_features (int): Number of input features
        out_features (int): Number of output features
        
    Learnable Parameters:
        weights: Standard neuron weights [out_features, in_features]
        bias: Standard neuron bias [out_features]
        p: Power parameter for F-mean [out_features]
        log_sigma: Gaussian width parameter [out_features]
        alphas: Three-way blending weights [out_features, 3]
    
    Example:
        >>> layer = HybridGaussianFMeanLayer(128, 256)
        >>> x = torch.randn(32, 128)  # batch_size=32
        >>> output = layer(x)  # shape: [32, 256]
    """
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.epsilon = 1e-8
        
        # Shared parameters
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Path-specific parameters
        self.p = nn.Parameter(torch.ones(out_features))  # F-Mean power parameter
        self.log_sigma = nn.Parameter(torch.zeros(out_features))  # Gaussian width
        self.alphas = nn.Parameter(torch.full((out_features, 3), 1/3))  # 3-way blending (initialized equally)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the three-way hybrid layer.
        
        Args:
            x: Input tensor of shape [batch_size, in_features]
            
        Returns:
            Output tensor of shape [batch_size, out_features]
        """
        # Path 1: Standard linear aggregation
        linear_out = F.linear(x, self.weights, self.bias)
        
        # Compute weighted inputs (shared across paths 2 and 3)
        z = x.unsqueeze(1) * self.weights  # Shape: [batch, out_features, in_features]
        
        # Path 2: F-Mean aggregation
        z_pos = F.softplus(z)
        p_un = self.p.view(1, -1, 1)
        z_pos_p = torch.pow(z_pos + self.epsilon, p_un)
        sum_z_pos_p = torch.sum(z_pos_p, dim=2, keepdim=True)
        agg_weights_fmean = z_pos_p / (sum_z_pos_p + self.epsilon)
        fmean_out = torch.sum(agg_weights_fmean * z, dim=2)
        
        # Path 3: Gaussian Support aggregation
        z_exp1 = z.unsqueeze(3)  # [batch, out_features, in_features, 1]
        z_exp2 = z.unsqueeze(2)  # [batch, out_features, 1, in_features]
        pairwise_dists = torch.sum((z_exp1 - z_exp2) ** 2, dim=1)  # Sum over out_features
        
        sigma = torch.exp(self.log_sigma).view(1, -1, 1)
        affinities = torch.exp(-pairwise_dists / (2 * sigma ** 2 + self.epsilon))
        
        agg_weights_gauss = F.softmax(affinities, dim=2)
        gaussian_out = torch.sum(agg_weights_gauss.unsqueeze(1) * z.unsqueeze(3), dim=2).squeeze(3)
        gaussian_out = torch.sum(gaussian_out, dim=2)
        
        # Three-way blending with softmax-normalized alphas
        # Ensures alpha_linear + alpha_fmean + alpha_gaussian = 1
        alpha_normalized = F.softmax(self.alphas, dim=1)  # Shape: [out_features, 3]
        
        # Blend outputs
        output = (alpha_normalized[:, 0].unsqueeze(0) * linear_out +
                  alpha_normalized[:, 1].unsqueeze(0) * fmean_out +
                  alpha_normalized[:, 2].unsqueeze(0) * gaussian_out)
        
        return output
    
    def get_alpha_distribution(self) -> torch.Tensor:
        """
        Returns the current distribution of blending weights.
        
        Returns:
            Tensor of shape [out_features, 3] with normalized alpha values
        """
        with torch.no_grad():
            return F.softmax(self.alphas, dim=1)
    
    def extra_repr(self) -> str:
        """String representation for printing the layer."""
        return f'in_features={self.in_features}, out_features={self.out_features}, paths=3'"""
Three-Way Hybrid Layer Implementation

This module implements a hybrid neuron layer that combines three aggregation
methods: linear, F-mean, and Gaussian support, with learnable blending weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HybridGaussianFMeanLayer(nn.Module):
    """
    Three-way hybrid layer combining linear, F-mean, and Gaussian aggregation.
    
    The layer blends three computational paths:
    1. Linear path: Standard weighted sum
    2. F-Mean path: Power-weighted aggregation
    3. Gaussian path: Affinity-based aggregation
    
    Parameters:
        in_features (int): Number of input features
        out_features (int): Number of output features
        
    Learnable Parameters:
        weights: Standard neuron weights [out_features, in_features]
        bias: Standard neuron bias [out_features]
        p: Power parameter for F-mean [out_features]
        log_sigma: Gaussian width parameter [out_features]
        alphas: Three-way blending weights [out_features, 3]
    
    Example:
        >>> layer = HybridGaussianFMeanLayer(128, 256)
        >>> x = torch.randn(32, 128)  # batch_size=32
        >>> output = layer(x)  # shape: [32, 256]
    """
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.epsilon = 1e-8
        
        # Shared parameters
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Path-specific parameters
        self.p = nn.Parameter(torch.ones(out_features))  # F-Mean power parameter
        self.log_sigma = nn.Parameter(torch.zeros(out_features))  # Gaussian width
        self.alphas = nn.Parameter(torch.full((out_features, 3), 1/3))  # 3-way blending (initialized equally)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the three-way hybrid layer.
        
        Args:
            x: Input tensor of shape [batch_size, in_features]
            
        Returns:
            Output tensor of shape [batch_size, out_features]
        """
        # Path 1: Standard linear aggregation
        linear_out = F.linear(x, self.weights, self.bias)
        
        # Compute weighted inputs (shared across paths 2 and 3)
        z = x.unsqueeze(1) * self.weights  # Shape: [batch, out_features, in_features]
        
        # Path 2: F-Mean aggregation
        z_pos = F.softplus(z)
        p_un = self.p.view(1, -1, 1)
        z_pos_p = torch.pow(z_pos + self.epsilon, p_un)
        sum_z_pos_p = torch.sum(z_pos_p, dim=2, keepdim=True)
        agg_weights_fmean = z_pos_p / (sum_z_pos_p + self.epsilon)
        fmean_out = torch.sum(agg_weights_fmean * z, dim=2)
        
        # Path 3: Gaussian Support aggregation
        z_exp1 = z.unsqueeze(3)  # [batch, out_features, in_features, 1]
        z_exp2 = z.unsqueeze(2)  # [batch, out_features, 1, in_features]
        pairwise_dists = torch.sum((z_exp1 - z_exp2) ** 2, dim=1)  # Sum over out_features
        
        sigma = torch.exp(self.log_sigma).view(1, -1, 1)
        affinities = torch.exp(-pairwise_dists / (2 * sigma ** 2 + self.epsilon))
        
        agg_weights_gauss = F.softmax(affinities, dim=2)
        gaussian_out = torch.sum(agg_weights_gauss.unsqueeze(1) * z.unsqueeze(3), dim=2).squeeze(3)
        gaussian_out = torch.sum(gaussian_out, dim=2)
        
        # Three-way blending with softmax-normalized alphas
        # Ensures alpha_linear + alpha_fmean + alpha_gaussian = 1
        alpha_normalized = F.softmax(self.alphas, dim=1)  # Shape: [out_features, 3]
        
        # Blend outputs
        output = (alpha_normalized[:, 0].unsqueeze(0) * linear_out +
                  alpha_normalized[:, 1].unsqueeze(0) * fmean_out +
                  alpha_normalized[:, 2].unsqueeze(0) * gaussian_out)
        
        return output
    
    def get_alpha_distribution(self) -> torch.Tensor:
        """
        Returns the current distribution of blending weights.
        
        Returns:
            Tensor of shape [out_features, 3] with normalized alpha values
        """
        with torch.no_grad():
            return F.softmax(self.alphas, dim=1)
    
    def extra_repr(self) -> str:
        """String representation for printing the layer."""
        return f'in_features={self.in_features}, out_features={self.out_features}, paths=3'"""
Three-Way Hybrid Layer Implementation

This module implements a hybrid neuron layer that combines three aggregation
methods: linear, F-mean, and Gaussian support, with learnable blending weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HybridGaussianFMeanLayer(nn.Module):
    """
    Three-way hybrid layer combining linear, F-mean, and Gaussian aggregation.
    
    The layer blends three computational paths:
    1. Linear path: Standard weighted sum
    2. F-Mean path: Power-weighted aggregation
    3. Gaussian path: Affinity-based aggregation
    
    Parameters:
        in_features (int): Number of input features
        out_features (int): Number of output features
        
    Learnable Parameters:
        weights: Standard neuron weights [out_features, in_features]
        bias: Standard neuron bias [out_features]
        p: Power parameter for F-mean [out_features]
        log_sigma: Gaussian width parameter [out_features]
        alphas: Three-way blending weights [out_features, 3]
    
    Example:
        >>> layer = HybridGaussianFMeanLayer(128, 256)
        >>> x = torch.randn(32, 128)  # batch_size=32
        >>> output = layer(x)  # shape: [32, 256]
    """
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.epsilon = 1e-8
        
        # Shared parameters
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Path-specific parameters
        self.p = nn.Parameter(torch.ones(out_features))  # F-Mean power parameter
        self.log_sigma = nn.Parameter(torch.zeros(out_features))  # Gaussian width
        self.alphas = nn.Parameter(torch.full((out_features, 3), 1/3))  # 3-way blending (initialized equally)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the three-way hybrid layer.
        
        Args:
            x: Input tensor of shape [batch_size, in_features]
            
        Returns:
            Output tensor of shape [batch_size, out_features]
        """
        # Path 1: Standard linear aggregation
        linear_out = F.linear(x, self.weights, self.bias)
        
        # Compute weighted inputs (shared across paths 2 and 3)
        z = x.unsqueeze(1) * self.weights  # Shape: [batch, out_features, in_features]
        
        # Path 2: F-Mean aggregation
        z_pos = F.softplus(z)
        p_un = self.p.view(1, -1, 1)
        z_pos_p = torch.pow(z_pos + self.epsilon, p_un)
        sum_z_pos_p = torch.sum(z_pos_p, dim=2, keepdim=True)
        agg_weights_fmean = z_pos_p / (sum_z_pos_p + self.epsilon)
        fmean_out = torch.sum(agg_weights_fmean * z, dim=2)
        
        # Path 3: Gaussian Support aggregation
        z_exp1 = z.unsqueeze(3)  # [batch, out_features, in_features, 1]
        z_exp2 = z.unsqueeze(2)  # [batch, out_features, 1, in_features]
        pairwise_dists = torch.sum((z_exp1 - z_exp2) ** 2, dim=1)  # Sum over out_features
        
        sigma = torch.exp(self.log_sigma).view(1, -1, 1)
        affinities = torch.exp(-pairwise_dists / (2 * sigma ** 2 + self.epsilon))
        
        agg_weights_gauss = F.softmax(affinities, dim=2)
        gaussian_out = torch.sum(agg_weights_gauss.unsqueeze(1) * z.unsqueeze(3), dim=2).squeeze(3)
        gaussian_out = torch.sum(gaussian_out, dim=2)
        
        # Three-way blending with softmax-normalized alphas
        # Ensures alpha_linear + alpha_fmean + alpha_gaussian = 1
        alpha_normalized = F.softmax(self.alphas, dim=1)  # Shape: [out_features, 3]
        
        # Blend outputs
        output = (alpha_normalized[:, 0].unsqueeze(0) * linear_out +
                  alpha_normalized[:, 1].unsqueeze(0) * fmean_out +
                  alpha_normalized[:, 2].unsqueeze(0) * gaussian_out)
        
        return output
    
    def get_alpha_distribution(self) -> torch.Tensor:
        """
        Returns the current distribution of blending weights.
        
        Returns:
            Tensor of shape [out_features, 3] with normalized alpha values
        """
        with torch.no_grad():
            return F.softmax(self.alphas, dim=1)
    
    def extra_repr(self) -> str:
        """String representation for printing the layer."""
        return f'in_features={self.in_features}, out_features={self.out_features}, paths=3'"""
Three-Way Hybrid Layer Implementation

This module implements a hybrid neuron layer that combines three aggregation
methods: linear, F-mean, and Gaussian support, with learnable blending weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HybridGaussianFMeanLayer(nn.Module):
    """
    Three-way hybrid layer combining linear, F-mean, and Gaussian aggregation.
    
    The layer blends three computational paths:
    1. Linear path: Standard weighted sum
    2. F-Mean path: Power-weighted aggregation
    3. Gaussian path: Affinity-based aggregation
    
    Parameters:
        in_features (int): Number of input features
        out_features (int): Number of output features
        
    Learnable Parameters:
        weights: Standard neuron weights [out_features, in_features]
        bias: Standard neuron bias [out_features]
        p: Power parameter for F-mean [out_features]
        log_sigma: Gaussian width parameter [out_features]
        alphas: Three-way blending weights [out_features, 3]
    
    Example:
        >>> layer = HybridGaussianFMeanLayer(128, 256)
        >>> x = torch.randn(32, 128)  # batch_size=32
        >>> output = layer(x)  # shape: [32, 256]
    """
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.epsilon = 1e-8
        
        # Shared parameters
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Path-specific parameters
        self.p = nn.Parameter(torch.ones(out_features))  # F-Mean power parameter
        self.log_sigma = nn.Parameter(torch.zeros(out_features))  # Gaussian width
        self.alphas = nn.Parameter(torch.full((out_features, 3), 1/3))  # 3-way blending (initialized equally)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the three-way hybrid layer.
        
        Args:
            x: Input tensor of shape [batch_size, in_features]
            
        Returns:
            Output tensor of shape [batch_size, out_features]
        """
        # Path 1: Standard linear aggregation
        linear_out = F.linear(x, self.weights, self.bias)
        
        # Compute weighted inputs (shared across paths 2 and 3)
        z = x.unsqueeze(1) * self.weights  # Shape: [batch, out_features, in_features]
        
        # Path 2: F-Mean aggregation
        z_pos = F.softplus(z)
        p_un = self.p.view(1, -1, 1)
        z_pos_p = torch.pow(z_pos + self.epsilon, p_un)
        sum_z_pos_p = torch.sum(z_pos_p, dim=2, keepdim=True)
        agg_weights_fmean = z_pos_p / (sum_z_pos_p + self.epsilon)
        fmean_out = torch.sum(agg_weights_fmean * z, dim=2)
        
        # Path 3: Gaussian Support aggregation
        z_exp1 = z.unsqueeze(3)  # [batch, out_features, in_features, 1]
        z_exp2 = z.unsqueeze(2)  # [batch, out_features, 1, in_features]
        pairwise_dists = torch.sum((z_exp1 - z_exp2) ** 2, dim=1)  # Sum over out_features
        
        sigma = torch.exp(self.log_sigma).view(1, -1, 1)
        affinities = torch.exp(-pairwise_dists / (2 * sigma ** 2 + self.epsilon))
        
        agg_weights_gauss = F.softmax(affinities, dim=2)
        gaussian_out = torch.sum(agg_weights_gauss.unsqueeze(1) * z.unsqueeze(3), dim=2).squeeze(3)
        gaussian_out = torch.sum(gaussian_out, dim=2)
        
        # Three-way blending with softmax-normalized alphas
        # Ensures alpha_linear + alpha_fmean + alpha_gaussian = 1
        alpha_normalized = F.softmax(self.alphas, dim=1)  # Shape: [out_features, 3]
        
        # Blend outputs
        output = (alpha_normalized[:, 0].unsqueeze(0) * linear_out +
                  alpha_normalized[:, 1].unsqueeze(0) * fmean_out +
                  alpha_normalized[:, 2].unsqueeze(0) * gaussian_out)
        
        return output
    
    def get_alpha_distribution(self) -> torch.Tensor:
        """
        Returns the current distribution of blending weights.
        
        Returns:
            Tensor of shape [out_features, 3] with normalized alpha values
        """
        with torch.no_grad():
            return F.softmax(self.alphas, dim=1)
    
    def extra_repr(self) -> str:
        """String representation for printing the layer."""
        return f'in_features={self.in_features}, out_features={self.out_features}, paths=3'"""
Three-Way Hybrid Layer Implementation

This module implements a hybrid neuron layer that combines three aggregation
methods: linear, F-mean, and Gaussian support, with learnable blending weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HybridGaussianFMeanLayer(nn.Module):
    """
    Three-way hybrid layer combining linear, F-mean, and Gaussian aggregation.
    
    The layer blends three computational paths:
    1. Linear path: Standard weighted sum
    2. F-Mean path: Power-weighted aggregation
    3. Gaussian path: Affinity-based aggregation
    
    Parameters:
        in_features (int): Number of input features
        out_features (int): Number of output features
        
    Learnable Parameters:
        weights: Standard neuron weights [out_features, in_features]
        bias: Standard neuron bias [out_features]
        p: Power parameter for F-mean [out_features]
        log_sigma: Gaussian width parameter [out_features]
        alphas: Three-way blending weights [out_features, 3]
    
    Example:
        >>> layer = HybridGaussianFMeanLayer(128, 256)
        >>> x = torch.randn(32, 128)  # batch_size=32
        >>> output = layer(x)  # shape: [32, 256]
    """
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.epsilon = 1e-8
        
        # Shared parameters
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Path-specific parameters
        self.p = nn.Parameter(torch.ones(out_features))  # F-Mean power parameter
        self.log_sigma = nn.Parameter(torch.zeros(out_features))  # Gaussian width
        self.alphas = nn.Parameter(torch.full((out_features, 3), 1/3))  # 3-way blending (initialized equally)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the three-way hybrid layer.
        
        Args:
            x: Input tensor of shape [batch_size, in_features]
            
        Returns:
            Output tensor of shape [batch_size, out_features]
        """
        # Path 1: Standard linear aggregation
        linear_out = F.linear(x, self.weights, self.bias)
        
        # Compute weighted inputs (shared across paths 2 and 3)
        z = x.unsqueeze(1) * self.weights  # Shape: [batch, out_features, in_features]
        
        # Path 2: F-Mean aggregation
        z_pos = F.softplus(z)
        p_un = self.p.view(1, -1, 1)
        z_pos_p = torch.pow(z_pos + self.epsilon, p_un)
        sum_z_pos_p = torch.sum(z_pos_p, dim=2, keepdim=True)
        agg_weights_fmean = z_pos_p / (sum_z_pos_p + self.epsilon)
        fmean_out = torch.sum(agg_weights_fmean * z, dim=2)
        
        # Path 3: Gaussian Support aggregation
        z_exp1 = z.unsqueeze(3)  # [batch, out_features, in_features, 1]
        z_exp2 = z.unsqueeze(2)  # [batch, out_features, 1, in_features]
        pairwise_dists = torch.sum((z_exp1 - z_exp2) ** 2, dim=1)  # Sum over out_features
        
        sigma = torch.exp(self.log_sigma).view(1, -1, 1)
        affinities = torch.exp(-pairwise_dists / (2 * sigma ** 2 + self.epsilon))
        
        agg_weights_gauss = F.softmax(affinities, dim=2)
        gaussian_out = torch.sum(agg_weights_gauss.unsqueeze(1) * z.unsqueeze(3), dim=2).squeeze(3)
        gaussian_out = torch.sum(gaussian_out, dim=2)
        
        # Three-way blending with softmax-normalized alphas
        # Ensures alpha_linear + alpha_fmean + alpha_gaussian = 1
        alpha_normalized = F.softmax(self.alphas, dim=1)  # Shape: [out_features, 3]
        
        # Blend outputs
        output = (alpha_normalized[:, 0].unsqueeze(0) * linear_out +
                  alpha_normalized[:, 1].unsqueeze(0) * fmean_out +
                  alpha_normalized[:, 2].unsqueeze(0) * gaussian_out)
        
        return output
    
    def get_alpha_distribution(self) -> torch.Tensor:
        """
        Returns the current distribution of blending weights.
        
        Returns:
            Tensor of shape [out_features, 3] with normalized alpha values
        """
        with torch.no_grad():
            return F.softmax(self.alphas, dim=1)
    
    def extra_repr(self) -> str:
        """String representation for printing the layer."""
        return f'in_features={self.in_features}, out_features={self.out_features}, paths=3'"""
Three-Way Hybrid Layer Implementation

This module implements a hybrid neuron layer that combines three aggregation
methods: linear, F-mean, and Gaussian support, with learnable blending weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HybridGaussianFMeanLayer(nn.Module):
    """
    Three-way hybrid layer combining linear, F-mean, and Gaussian aggregation.
    
    The layer blends three computational paths:
    1. Linear path: Standard weighted sum
    2. F-Mean path: Power-weighted aggregation
    3. Gaussian path: Affinity-based aggregation
    
    Parameters:
        in_features (int): Number of input features
        out_features (int): Number of output features
        
    Learnable Parameters:
        weights: Standard neuron weights [out_features, in_features]
        bias: Standard neuron bias [out_features]
        p: Power parameter for F-mean [out_features]
        log_sigma: Gaussian width parameter [out_features]
        alphas: Three-way blending weights [out_features, 3]
    
    Example:
        >>> layer = HybridGaussianFMeanLayer(128, 256)
        >>> x = torch.randn(32, 128)  # batch_size=32
        >>> output = layer(x)  # shape: [32, 256]
    """
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.epsilon = 1e-8
        
        # Shared parameters
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Path-specific parameters
        self.p = nn.Parameter(torch.ones(out_features))  # F-Mean power parameter
        self.log_sigma = nn.Parameter(torch.zeros(out_features))  # Gaussian width
        self.alphas = nn.Parameter(torch.full((out_features, 3), 1/3))  # 3-way blending (initialized equally)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the three-way hybrid layer.
        
        Args:
            x: Input tensor of shape [batch_size, in_features]
            
        Returns:
            Output tensor of shape [batch_size, out_features]
        """
        # Path 1: Standard linear aggregation
        linear_out = F.linear(x, self.weights, self.bias)
        
        # Compute weighted inputs (shared across paths 2 and 3)
        z = x.unsqueeze(1) * self.weights  # Shape: [batch, out_features, in_features]
        
        # Path 2: F-Mean aggregation
        z_pos = F.softplus(z)
        p_un = self.p.view(1, -1, 1)
        z_pos_p = torch.pow(z_pos + self.epsilon, p_un)
        sum_z_pos_p = torch.sum(z_pos_p, dim=2, keepdim=True)
        agg_weights_fmean = z_pos_p / (sum_z_pos_p + self.epsilon)
        fmean_out = torch.sum(agg_weights_fmean * z, dim=2)
        
        # Path 3: Gaussian Support aggregation
        z_exp1 = z.unsqueeze(3)  # [batch, out_features, in_features, 1]
        z_exp2 = z.unsqueeze(2)  # [batch, out_features, 1, in_features]
        pairwise_dists = torch.sum((z_exp1 - z_exp2) ** 2, dim=1)  # Sum over out_features
        
        sigma = torch.exp(self.log_sigma).view(1, -1, 1)
        affinities = torch.exp(-pairwise_dists / (2 * sigma ** 2 + self.epsilon))
        
        agg_weights_gauss = F.softmax(affinities, dim=2)
        gaussian_out = torch.sum(agg_weights_gauss.unsqueeze(1) * z.unsqueeze(3), dim=2).squeeze(3)
        gaussian_out = torch.sum(gaussian_out, dim=2)
        
        # Three-way blending with softmax-normalized alphas
        # Ensures alpha_linear + alpha_fmean + alpha_gaussian = 1
        alpha_normalized = F.softmax(self.alphas, dim=1)  # Shape: [out_features, 3]
        
        # Blend outputs
        output = (alpha_normalized[:, 0].unsqueeze(0) * linear_out +
                  alpha_normalized[:, 1].unsqueeze(0) * fmean_out +
                  alpha_normalized[:, 2].unsqueeze(0) * gaussian_out)
        
        return output
    
    def get_alpha_distribution(self) -> torch.Tensor:
        """
        Returns the current distribution of blending weights.
        
        Returns:
            Tensor of shape [out_features, 3] with normalized alpha values
        """
        with torch.no_grad():
            return F.softmax(self.alphas, dim=1)
    
    def extra_repr(self) -> str:
        """String representation for printing the layer."""
        return f'in_features={self.in_features}, out_features={self.out_features}, paths=3'"""
Three-Way Hybrid Layer Implementation

This module implements a hybrid neuron layer that combines three aggregation
methods: linear, F-mean, and Gaussian support, with learnable blending weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HybridGaussianFMeanLayer(nn.Module):
    """
    Three-way hybrid layer combining linear, F-mean, and Gaussian aggregation.
    
    The layer blends three computational paths:
    1. Linear path: Standard weighted sum
    2. F-Mean path: Power-weighted aggregation
    3. Gaussian path: Affinity-based aggregation
    
    Parameters:
        in_features (int): Number of input features
        out_features (int): Number of output features
        
    Learnable Parameters:
        weights: Standard neuron weights [out_features, in_features]
        bias: Standard neuron bias [out_features]
        p: Power parameter for F-mean [out_features]
        log_sigma: Gaussian width parameter [out_features]
        alphas: Three-way blending weights [out_features, 3]
    
    Example:
        >>> layer = HybridGaussianFMeanLayer(128, 256)
        >>> x = torch.randn(32, 128)  # batch_size=32
        >>> output = layer(x)  # shape: [32, 256]
    """
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.epsilon = 1e-8
        
        # Shared parameters
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Path-specific parameters
        self.p = nn.Parameter(torch.ones(out_features))  # F-Mean power parameter
        self.log_sigma = nn.Parameter(torch.zeros(out_features))  # Gaussian width
        self.alphas = nn.Parameter(torch.full((out_features, 3), 1/3))  # 3-way blending (initialized equally)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the three-way hybrid layer.
        
        Args:
            x: Input tensor of shape [batch_size, in_features]
            
        Returns:
            Output tensor of shape [batch_size, out_features]
        """
        # Path 1: Standard linear aggregation
        linear_out = F.linear(x, self.weights, self.bias)
        
        # Compute weighted inputs (shared across paths 2 and 3)
        z = x.unsqueeze(1) * self.weights  # Shape: [batch, out_features, in_features]
        
        # Path 2: F-Mean aggregation
        z_pos = F.softplus(z)
        p_un = self.p.view(1, -1, 1)
        z_pos_p = torch.pow(z_pos + self.epsilon, p_un)
        sum_z_pos_p = torch.sum(z_pos_p, dim=2, keepdim=True)
        agg_weights_fmean = z_pos_p / (sum_z_pos_p + self.epsilon)
        fmean_out = torch.sum(agg_weights_fmean * z, dim=2)
        
        # Path 3: Gaussian Support aggregation
        z_exp1 = z.unsqueeze(3)  # [batch, out_features, in_features, 1]
        z_exp2 = z.unsqueeze(2)  # [batch, out_features, 1, in_features]
        pairwise_dists = torch.sum((z_exp1 - z_exp2) ** 2, dim=1)  # Sum over out_features
        
        sigma = torch.exp(self.log_sigma).view(1, -1, 1)
        affinities = torch.exp(-pairwise_dists / (2 * sigma ** 2 + self.epsilon))
        
        agg_weights_gauss = F.softmax(affinities, dim=2)
        gaussian_out = torch.sum(agg_weights_gauss.unsqueeze(1) * z.unsqueeze(3), dim=2).squeeze(3)
        gaussian_out = torch.sum(gaussian_out, dim=2)
        
        # Three-way blending with softmax-normalized alphas
        # Ensures alpha_linear + alpha_fmean + alpha_gaussian = 1
        alpha_normalized = F.softmax(self.alphas, dim=1)  # Shape: [out_features, 3]
        
        # Blend outputs
        output = (alpha_normalized[:, 0].unsqueeze(0) * linear_out +
                  alpha_normalized[:, 1].unsqueeze(0) * fmean_out +
                  alpha_normalized[:, 2].unsqueeze(0) * gaussian_out)
        
        return output
    
    def get_alpha_distribution(self) -> torch.Tensor:
        """
        Returns the current distribution of blending weights.
        
        Returns:
            Tensor of shape [out_features, 3] with normalized alpha values
        """
        with torch.no_grad():
            return F.softmax(self.alphas, dim=1)
    
    def extra_repr(self) -> str:
        """String representation for printing the layer."""
        return f'in_features={self.in_features}, out_features={self.out_features}, paths=3'"""
Three-Way Hybrid Layer Implementation

This module implements a hybrid neuron layer that combines three aggregation
methods: linear, F-mean, and Gaussian support, with learnable blending weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HybridGaussianFMeanLayer(nn.Module):
    """
    Three-way hybrid layer combining linear, F-mean, and Gaussian aggregation.
    
    The layer blends three computational paths:
    1. Linear path: Standard weighted sum
    2. F-Mean path: Power-weighted aggregation
    3. Gaussian path: Affinity-based aggregation
    
    Parameters:
        in_features (int): Number of input features
        out_features (int): Number of output features
        
    Learnable Parameters:
        weights: Standard neuron weights [out_features, in_features]
        bias: Standard neuron bias [out_features]
        p: Power parameter for F-mean [out_features]
        log_sigma: Gaussian width parameter [out_features]
        alphas: Three-way blending weights [out_features, 3]
    
    Example:
        >>> layer = HybridGaussianFMeanLayer(128, 256)
        >>> x = torch.randn(32, 128)  # batch_size=32
        >>> output = layer(x)  # shape: [32, 256]
    """
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.epsilon = 1e-8
        
        # Shared parameters
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Path-specific parameters
        self.p = nn.Parameter(torch.ones(out_features))  # F-Mean power parameter
        self.log_sigma = nn.Parameter(torch.zeros(out_features))  # Gaussian width
        self.alphas = nn.Parameter(torch.full((out_features, 3), 1/3))  # 3-way blending (initialized equally)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the three-way hybrid layer.
        
        Args:
            x: Input tensor of shape [batch_size, in_features]
            
        Returns:
            Output tensor of shape [batch_size, out_features]
        """
        # Path 1: Standard linear aggregation
        linear_out = F.linear(x, self.weights, self.bias)
        
        # Compute weighted inputs (shared across paths 2 and 3)
        z = x.unsqueeze(1) * self.weights  # Shape: [batch, out_features, in_features]
        
        # Path 2: F-Mean aggregation
        z_pos = F.softplus(z)
        p_un = self.p.view(1, -1, 1)
        z_pos_p = torch.pow(z_pos + self.epsilon, p_un)
        sum_z_pos_p = torch.sum(z_pos_p, dim=2, keepdim=True)
        agg_weights_fmean = z_pos_p / (sum_z_pos_p + self.epsilon)
        fmean_out = torch.sum(agg_weights_fmean * z, dim=2)
        
        # Path 3: Gaussian Support aggregation
        z_exp1 = z.unsqueeze(3)  # [batch, out_features, in_features, 1]
        z_exp2 = z.unsqueeze(2)  # [batch, out_features, 1, in_features]
        pairwise_dists = torch.sum((z_exp1 - z_exp2) ** 2, dim=1)  # Sum over out_features
        
        sigma = torch.exp(self.log_sigma).view(1, -1, 1)
        affinities = torch.exp(-pairwise_dists / (2 * sigma ** 2 + self.epsilon))
        
        agg_weights_gauss = F.softmax(affinities, dim=2)
        gaussian_out = torch.sum(agg_weights_gauss.unsqueeze(1) * z.unsqueeze(3), dim=2).squeeze(3)
        gaussian_out = torch.sum(gaussian_out, dim=2)
        
        # Three-way blending with softmax-normalized alphas
        # Ensures alpha_linear + alpha_fmean + alpha_gaussian = 1
        alpha_normalized = F.softmax(self.alphas, dim=1)  # Shape: [out_features, 3]
        
        # Blend outputs
        output = (alpha_normalized[:, 0].unsqueeze(0) * linear_out +
                  alpha_normalized[:, 1].unsqueeze(0) * fmean_out +
                  alpha_normalized[:, 2].unsqueeze(0) * gaussian_out)
        
        return output
    
    def get_alpha_distribution(self) -> torch.Tensor:
        """
        Returns the current distribution of blending weights.
        
        Returns:
            Tensor of shape [out_features, 3] with normalized alpha values
        """
        with torch.no_grad():
            return F.softmax(self.alphas, dim=1)
    
    def extra_repr(self) -> str:
        """String representation for printing the layer."""
        return f'in_features={self.in_features}, out_features={self.out_features}, paths=3'