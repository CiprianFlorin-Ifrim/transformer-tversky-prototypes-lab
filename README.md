# Tversky Neural Networks: Implementation and Experimental Analysis

A PyTorch implementation of Tversky Neural Networks based on Tversky's contrast model for similarity computation, with comprehensive experiments and TVM microcontroller deployment validation.

## Overview

This repository implements and evaluates **Tversky Neural Networks**, a novel architecture that replaces traditional dot-product similarity with Tversky's asymmetric similarity function. Unlike conventional neural networks that rely on geometric similarity, Tversky layers explicitly model feature presence/absence through learnable feature representations.

## Background

### Tversky Similarity Theory

Tversky's contrast model measures similarity between objects A and B based on:
- **Common features**: f(A ∩ B)
- **Distinctive features of A**: f(A - B)  
- **Distinctive features of B**: f(B - A)

The similarity formula is:
```
S(A,B) = θf(A∩B) - αf(A-B) - βf(B-A)
```

### Key Advantages

1. **Asymmetric**: S(A,B) ≠ S(B,A) in general, capturing directional relationships
2. **Feature-based**: Uses explicit feature representations rather than geometric similarity
3. **Learnable**: θ, α, β parameters can be optimized end-to-end
4. **Interpretable**: Feature banks provide insight into learned representations

## Repository Contents

### Core Implementation

- **`TverskyProjectionLayer`**: PyTorch implementation of the Tversky similarity layer
- **Quantization Pipeline**: Complete workflow for microcontroller deployment via TVM
- **Experimental Framework**: Comprehensive testing on synthetic and real-world datasets

### Experiments

1. **XOR Classification**: Tests non-linear separability with varying feature counts
2. **Biology Q&A Dataset**: Real-world language modeling evaluation
3. **TVM Validation**: Quantization and microcontroller compilation pipeline

## Quick Start

### Requirements

```bash
# Core dependencies
torch>=2.0.0
numpy
matplotlib
pandas
tqdm

# For biology dataset experiments (optional)
# Your custom vocabulary files and biology Q&A dataset
```

### Basic Usage

```python
import torch
from tversky_layer import TverskyProjectionLayer

# Create a Tversky layer
layer = TverskyProjectionLayer(
    in_features=128,      # Input dimension
    out_features=64,      # Number of prototypes
    num_features=32       # Size of feature bank
)

# Forward pass
input_tensor = torch.randn(10, 128)  # Batch of 10 samples
similarities = layer(input_tensor)   # Output: (10, 64)
```

### Running Experiments

```python
# Run all experiments
python tversky_experiments.py

# Run only XOR experiment
from tversky_experiments import run_xor_experiment
results = run_xor_experiment()

# Run biology Q&A experiment (requires data files)
from tversky_experiments import run_biology_experiment
bio_results = run_biology_experiment(vocab_data, char_to_id, csv_path)
```

## Experimental Results

### XOR Classification

The XOR problem tests the layer's ability to handle non-linearly separable data:

| Features | Accuracy |
|----------|----------|
| 1        | 50%      |
| 2        | 50%      |
| 4        | 75%      |
| 8        | 75%      |
| 16       | 75%      |
| 32       | 100%     |
| 64       | 100%     |

**Key Finding**: Performance improves with feature count up to a threshold, then plateaus.

### Biology Q&A Dataset

Real-world language modeling task shows:
- Consistent improvement with more features
- Optimal performance around 64-96 features
- Demonstrates practical applicability to NLP tasks

## Implementation Details

### Layer Architecture

```python
class TverskyProjectionLayer(nn.Module):
    def __init__(self, in_features, out_features, num_features):
        # Learnable prototype bank (P × D)
        self.prototypes = nn.Parameter(torch.randn(out_features, in_features))
        
        # Learnable feature bank (F × D)  
        self.features = nn.Parameter(torch.randn(num_features, in_features))
        
        # Tversky parameters
        self.theta = nn.Parameter(torch.randn(1))  # Common features weight
        self.alpha = nn.Parameter(torch.randn(1))  # Input distinctive weight
        self.beta = nn.Parameter(torch.randn(1))   # Prototype distinctive weight
```

### Key Features

- **Flexible similarity computation**: Supports multiple intersection/difference reduction methods
- **End-to-end differentiable**: All parameters learned via backpropagation
- **Scalable**: Handles both 2D and 3D input tensors (batch, sequence, features)
- **Memory efficient**: Optimized broadcasting operations

## File Structure

```
tversky-neural-networks/
├── README.md
├── tversky_experiments.ipynb          # Main experimental notebook
├── tvm_validation.ipynb               # TVM quantization pipeline
├── requirements.txt                   # Python dependencies
└── data/                              # Data files (not included)
    ├── vocabulary_cache.json          # Custom vocabulary
    └── biology_qa_3000_augmented.csv  # Biology Q&A dataset
```

## Research Applications

Tversky Neural Networks may be particularly useful for:

- **Similarity-based retrieval tasks**
- **Few-shot learning scenarios** 
- **Tasks requiring asymmetric relationships**
- **Interpretable feature learning**
- **Resource-constrained deployment** (via quantization)

## Citation

Based on the paper "Tversky Neural Networks" (arXiv:2506.11035v1).

```bibtex
@article{tversky_neural_networks,
  title={Tversky Neural Networks},
  author={[Authors]},
  journal={arXiv preprint arXiv:2506.11035},
  year={2024}
}
```
