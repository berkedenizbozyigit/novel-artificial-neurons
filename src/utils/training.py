"""
Training and Evaluation Utilities

This module provides functions for training and testing neural networks
with support for hybrid layers and parameter monitoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
from tqdm import tqdm


def train_epoch(
    model: nn.Module,
    device: torch.device,
    train_loader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    log_interval: int = 100,
    gradient_clip: Optional[float] = None
) -> float:
    """
    Train the model for one epoch.
    
    Parameters:
        model: Neural network model
        device: Device to train on (cuda/cpu)
        train_loader: Training data loader
        optimizer: Optimizer
        epoch: Current epoch number
        log_interval: How often to log training status
        gradient_clip: Value to clip gradients (None = no clipping)
    
    Returns:
        Average training loss for the epoch
    
    Example:
        >>> model = HybridCNN(HybridFMeanLayer)
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> loss = train_epoch(model, device, train_loader, optimizer, epoch=1)
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        
        # Gradient clipping if specified
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Log progress
        if batch_idx % log_interval == 0:
            print(f'  Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    avg_loss = total_loss / num_batches
    return avg_loss


def test(
    model: nn.Module,
    device: torch.device,
    test_loader,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate the model on test data.
    
    Parameters:
        model: Neural network model
        device: Device to evaluate on (cuda/cpu)
        test_loader: Test data loader
        verbose: Whether to print results
    
    Returns:
        Dictionary containing 'loss' and 'accuracy'
    
    Example:
        >>> model = HybridCNN(HybridFMeanLayer)
        >>> results = test(model, device, test_loader)
        >>> print(f"Test Accuracy: {results['accuracy']:.2f}%")
    """
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    if verbose:
        print(f'  Test set: Average loss: {test_loss:.4f}, '
              f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    
    return {
        'loss': test_loss,
        'accuracy': accuracy
    }


def get_hybrid_parameters(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract hybrid layer parameters (alpha, p, sigma) for monitoring.
    
    Parameters:
        model: Neural network model with hybrid layers
    
    Returns:
        Dictionary of parameter names to their values
    
    Example:
        >>> model = HybridCNN(HybridFMeanLayer)
        >>> params = get_hybrid_parameters(model)
        >>> print(f"Mean alpha: {params.get('alpha', [0])[0]:.3f}")
    """
    params = {}
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'alpha' in name and 'alphas' not in name:
                # Two-way hybrid: sigmoid activation
                params['alpha'] = torch.sigmoid(param).cpu()
            elif 'alphas' in name:
                # Three-way hybrid: softmax activation
                params['alphas'] = F.softmax(param, dim=1).cpu()
            elif 'log_p' in name or 'p' in name:
                if 'log_p' in name:
                    params['p'] = torch.exp(param).cpu()
                else:
                    params['p'] = param.cpu()
            elif 'log_sigma' in name:
                params['sigma'] = torch.exp(param).cpu()
    
    return params


def print_hybrid_parameters(model: nn.Module, verbose: bool = True):
    """
    Print current values of hybrid layer parameters.
    
    Parameters:
        model: Neural network model with hybrid layers
        verbose: Whether to print detailed information
    
    Example:
        >>> model = HybridCNN(HybridFMeanLayer)
        >>> print_hybrid_parameters(model)
    """
    params = get_hybrid_parameters(model)
    
    if not params:
        if verbose:
            print("  No hybrid parameters found in model.")
        return
    
    print("  --- Hybrid Parameter Analysis ---")
    
    # Alpha parameters (blending weights)
    if 'alpha' in params:
        alpha_mean = params['alpha'].mean().item()
        print(f"  --> Mean alpha value (novelty blend): {alpha_mean:.4f}")
    
    if 'alphas' in params:
        alpha_mean = params['alphas'].mean(dim=0)
        print(f"  --> Mean Alphas (Linear | F-Mean | Gaussian): "
              f"{alpha_mean[0]:.3f} | {alpha_mean[1]:.3f} | {alpha_mean[2]:.3f}")
    
    # P parameter (F-Mean power)
    if 'p' in params:
        p_mean = params['p'].mean().item()
        print(f"  --> Mean p value: {p_mean:.4f}")
    
    # Sigma parameter (Gaussian width)
    if 'sigma' in params:
        sigma_mean = params['sigma'].mean().item()
        print(f"  --> Mean sigma value: {sigma_mean:.4f}")


def create_optimizer(
    model: nn.Module,
    lr_weights: float = 0.001,
    lr_params: float = 0.01,
    weight_decay: float = 0.0
) -> torch.optim.Optimizer:
    """
    Create optimizer with differential learning rates for hybrid parameters.
    
    Hybrid parameters (alpha, p, sigma) get higher learning rates to encourage
    exploration of novel aggregation strategies, while standard weights use
    conservative rates for stability.
    
    Parameters:
        model: Neural network model
        lr_weights: Learning rate for standard weights (default: 0.001)
        lr_params: Learning rate for hybrid parameters (default: 0.01, 10x higher)
        weight_decay: L2 regularization factor (default: 0.0)
    
    Returns:
        Adam optimizer with parameter groups
    
    Example:
        >>> model = HybridCNN(HybridFMeanLayer)
        >>> optimizer = create_optimizer(model, lr_weights=0.001, lr_params=0.01)
    """
    # Separate hybrid parameters from standard parameters
    hybrid_params = []
    standard_params = []
    
    for name, param in model.named_parameters():
        if any(novel_name in name for novel_name in ['p', 'log_p', 'alpha', 'log_sigma', 'alphas']):
            hybrid_params.append(param)
        else:
            standard_params.append(param)
    
    # Create optimizer with parameter groups
    optimizer = torch.optim.Adam([
        {'params': standard_params, 'lr': lr_weights, 'weight_decay': weight_decay},
        {'params': hybrid_params, 'lr': lr_params, 'weight_decay': 0.0}  # No weight decay on hybrid params
    ])
    
    print(f"\nOptimizer configured with differential learning rates:")
    print(f"  Standard parameters: {len(standard_params)} params, lr={lr_weights}")
    print(f"  Hybrid parameters: {len(hybrid_params)} params, lr={lr_params} (10x)")
    
    return optimizer


def train_model(
    model: nn.Module,
    device: torch.device,
    train_loader,
    test_loader,
    epochs: int = 10,
    lr_weights: float = 0.001,
    lr_params: float = 0.01,
    gradient_clip: Optional[float] = 1.0,
    early_stopping_patience: int = 10,
    verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Complete training loop with early stopping and parameter monitoring.
    
    Parameters:
        model: Neural network model
        device: Device to train on
        train_loader: Training data loader
        test_loader: Test data loader
        epochs: Maximum number of epochs
        lr_weights: Learning rate for standard parameters
        lr_params: Learning rate for hybrid parameters
        gradient_clip: Gradient clipping value (None = no clipping)
        early_stopping_patience: Epochs without improvement before stopping
        verbose: Whether to print training progress
    
    Returns:
        Dictionary containing training history
    
    Example:
        >>> model = HybridCNN(HybridFMeanLayer).to(device)
        >>> history = train_model(model, device, train_loader, test_loader, epochs=50)
    """
    # Create optimizer with differential learning rates
    optimizer = create_optimizer(model, lr_weights, lr_params)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    # Track history
    history = {
        'train_loss': [],
        'test_loss': [],
        'test_accuracy': [],
    }
    
    best_accuracy = 0.0
    epochs_no_improve = 0
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_epoch(model, device, train_loader, optimizer, epoch, gradient_clip=gradient_clip)
        
        # Test
        test_results = test(model, device, test_loader, verbose=verbose)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_results['loss'])
        history['test_accuracy'].append(test_results['accuracy'])
        
        # Print hybrid parameters
        if verbose:
            print_hybrid_parameters(model)
        
        # Learning rate scheduling
        scheduler.step(test_results['accuracy'])
        
        # Early stopping
        if test_results['accuracy'] > best_accuracy:
            best_accuracy = test_results['accuracy']
            epochs_no_improve = 0
            if verbose:
                print(f"  --> New best accuracy: {best_accuracy:.2f}%")
        else:
            epochs_no_improve += 1
            if verbose:
                print(f"  --> No improvement for {epochs_no_improve} epochs.")
        
        if epochs_no_improve >= early_stopping_patience:
            if verbose:
                print(f"\nEarly stopping after {epoch} epochs. Best accuracy: {best_accuracy:.2f}%")
            break
    
    return history