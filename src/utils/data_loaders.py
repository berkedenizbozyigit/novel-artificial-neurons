"""
Data Loading Utilities for CIFAR-10

This module provides data loaders for CIFAR-10 dataset with optional
Gaussian noise augmentation for robustness testing.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class AddGaussianNoise:
    """
    Transform to add Gaussian noise to images for robustness testing.
    
    Parameters:
        mean (float): Mean of the Gaussian noise (default: 0.0)
        std (float): Standard deviation of the noise (default: 0.1)
    
    Example:
        >>> noise_transform = AddGaussianNoise(mean=0., std=0.1)
        >>> noisy_image = noise_transform(image)
    """
    
    def __init__(self, mean: float = 0.0, std: float = 0.1):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to the input tensor.
        
        Args:
            tensor: Input image tensor
            
        Returns:
            Noisy image tensor
        """
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std})'


def get_cifar10_loaders(
    batch_size: int = 128,
    use_noise: bool = False,
    noise_std: float = 0.1,
    num_workers: int = 2,
    download: bool = True
):
    """
    Get CIFAR-10 train and test data loaders.
    
    Parameters:
        batch_size (int): Batch size for data loaders (default: 128)
        use_noise (bool): Whether to add Gaussian noise (default: False)
        noise_std (float): Standard deviation of noise if used (default: 0.1)
        num_workers (int): Number of worker processes for data loading (default: 2)
        download (bool): Whether to download CIFAR-10 if not present (default: True)
    
    Returns:
        tuple: (train_loader, test_loader)
    
    Example:
        >>> # Standard CIFAR-10
        >>> train_loader, test_loader = get_cifar10_loaders(batch_size=128)
        >>> 
        >>> # Noisy CIFAR-10 for robustness testing
        >>> noisy_train, noisy_test = get_cifar10_loaders(
        ...     batch_size=128, 
        ...     use_noise=True, 
        ...     noise_std=0.1
        ... )
    """
    
    # Base transforms: convert to tensor and normalize
    base_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    
    # Add noise if requested (for robustness testing)
    if use_noise:
        base_transforms.append(AddGaussianNoise(mean=0.0, std=noise_std))
    
    transform = transforms.Compose(base_transforms)
    
    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=download,
        transform=transform
    )
    
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=download,
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def get_dataset_info(loader: DataLoader) -> dict:
    """
    Get information about a dataset from its data loader.
    
    Parameters:
        loader: PyTorch DataLoader
    
    Returns:
        dict: Dataset information including size, batch size, num batches
    
    Example:
        >>> train_loader, _ = get_cifar10_loaders()
        >>> info = get_dataset_info(train_loader)
        >>> print(f"Dataset size: {info['dataset_size']}")
    """
    dataset = loader.dataset
    
    info = {
        'dataset_size': len(dataset),
        'batch_size': loader.batch_size,
        'num_batches': len(loader),
        'num_workers': loader.num_workers,
    }
    
    # Get sample to determine shapes
    sample_data, sample_label = dataset[0]
    info['data_shape'] = tuple(sample_data.shape)
    info['num_classes'] = len(set([dataset[i][1] for i in range(min(1000, len(dataset)))]))
    
    return info