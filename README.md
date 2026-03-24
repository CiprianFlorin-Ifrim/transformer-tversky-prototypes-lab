# Tversky Neural Networks

PyTorch implementation and experimental analysis of Tversky Neural Networks based on Tversky's contrast model for similarity computation. Reference: [arXiv:2506.11035v1](https://arxiv.org/abs/2506.11035).

---

## Overview

Tversky's contrast model measures similarity between objects A and B based on common features f(A intersect B), features distinctive to A (f(A - B)), and features distinctive to B (f(B - A)):

```
S(A, B) = theta * f(A intersect B) - alpha * f(A - B) - beta * f(B - A)
```

This gives three properties standard dot-product similarity lacks: asymmetry (S(A,B) != S(B,A) in general), explicit feature-presence reasoning, and learnable theta/alpha/beta parameters that control how much each term contributes.

---

## Repository Structure

```
.
├── tversky_paper_replication.ipynb              # Paper implementation + XOR / biology Q&A experiments
├── tversky_transformer_investigation.ipynb      # Project ablation: is Tversky useful vs linear baselines?
├── requirements.txt                             # Python dependencies
└── data/                                        # Data files (not included)
    ├── vocabulary_cache.json                    # Custom vocabulary
    └── biology_qa_3000_augmented.csv
```

---

## Notebooks

**tversky_paper_replication.ipynb** implements the `TverskyProjectionLayer` from the paper and validates it on the paper's own benchmarks: XOR classification (non-linearly separable data) and biology Q&A language modeling. Covers the full layer implementation, flexible intersection/difference reduction modes, and a small transformer with a Tversky output head.

**tversky_transformer_investigation.ipynb** is a project-specific ablation that asks whether Tversky similarity is useful in practice compared to standard linear layers. Tests on synthetic directional classification and LM next-token prediction tasks, sweeps feature counts, benchmarks alternative linear layer variants (grouped, low-rank, block-diagonal, etc.) inside a mini-transformer, and measures storage and speed tradeoffs.


---

## Requirements

```
torch>=2.0.0
numpy
matplotlib
pandas
tqdm
```

---

## Citation

```
@article{tversky_neural_networks,
  title   = {Tversky Neural Networks},
  journal = {arXiv preprint arXiv:2506.11035},
  year    = {2025}
}
```
