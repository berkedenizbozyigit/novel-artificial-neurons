# A New Type of Artificial Neuron: Exploring Nonlinear Aggregation Functions

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> MSc Artificial Intelligence Dissertation Project  
> University of York, Department of Computer Science  
> September 2025

## 📖 Overview

This repository contains the implementation and evaluation of novel artificial neuron architectures that replace traditional weighted sum aggregation with learnable, nonlinear aggregation functions. The project explores two main approaches:

1. **F-Mean Neuron**: Uses learnable generalized F-mean aggregation with power parameter *p*
2. **Gaussian Support Neuron**: Employs pairwise Gaussian affinities between weighted inputs
3. **Three-Way Hybrid**: Combines both approaches with learnable blending weights

### Key Innovation

Traditional neurons use **sum aggregation** (equivalent to mean average). We generalize this by:
- Making aggregation functions **learnable** (with parameters like *p* and σ)
- Using **hybrid architectures** that blend novel and traditional paths
- Ensuring **numerical stability** through careful implementation

## 🎯 Research Questions

- Can alternative aggregation functions improve neural network performance?
- How do novel neurons behave under noisy conditions?
- What aggregation strategies do networks learn during training?

## 📊 Results Summary

| Model | Standard CIFAR-10 | Noisy CIFAR-10 |
|-------|------------------|----------------|
| **F-Mean Hybrid** | 55.17% | 53.56% |
| **Gaussian Hybrid** | 54.30% | 53.30% |
| **3-Way Hybrid** | **55.21%** | **54.72%** |

*The 3-Way Hybrid shows best performance, especially under noisy conditions*

## 📄 Full Dissertation

The complete dissertation is available here:
https://drive.google.com/file/d/1Fmmc6ZvoUDFI9vBJtXSbokAay0W25JAT/view?usp=sharing


## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/berkebzt/novel-artificial-neurons.git
cd novel-artificial-neurons

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

import torch
from src.layers import HybridFMeanLayer, HybridGaussianLayer
from src.models import HybridCNN
from src.utils import get_cifar10_loaders

# Load data
train_loader, test_loader = get_cifar10_loaders(batch_size=128)

# Create model with F-Mean hybrid neurons
model = HybridCNN(HybridFMeanLayer, projection_dim=256, hidden_dim=256)

# Train model (coming soon in notebooks/)
```
Key Findings
Parameter Evolution

Alpha (blend weight): 0.5 → 0.76 (F-Mean), 0.69 (Gaussian)

Networks prefer novel aggregation over standard linear


p parameter: 1.0 → 0.32

Learns sub-linear aggregation (more robust to outliers)


σ parameter: 1.0 → 3.59

Prefers broader, mean-like aggregation


Performance Insights

Noise Robustness: 3-Way Hybrid shows smallest performance drop
Complementary Mechanisms: F-Mean and Gaussian learn different strategies
Stable Training: Hybrid architecture prevents training failures

 Experimental Setup

Dataset: CIFAR-10 (50K train, 10K test)
Noise: Gaussian noise (μ=0, σ=0.1)
Architecture: CNN backbone + hybrid classifier
Training: 50 epochs, Adam optimizer, differential learning rates
Hardware: Google Colab A100 GPU

Contributing
Contributions welcome! Extension ideas:

Apply to other datasets (ImageNet, MNIST variants)
Explore other aggregation functions (Lehmer mean, trimmed mean)
Investigate implicit layer approach with p-norm minimization
Add theoretical analysis of gradient flow
Benchmark computational efficiency

📧 Contact
Berke Deniz Bozyigit
MSc Artificial Intelligence
University of York
rvp516@york.ac.uk

🙏 Acknowledgements

Prof. William Smith (Supervisor)
University of York, Department of Computer Science
