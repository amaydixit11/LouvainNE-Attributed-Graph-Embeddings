# ADR 0032: Logistic Regression for Embedding Evaluation

**Status**: Accepted

## Context
The goal of the project is to evaluate the quality of the unsupervised embeddings $\mathbf{Z}$.

## Decision
Use **$\ell_2$-regularized Logistic Regression** as the downstream classifier for node classification.

## Rationale
- **Linearity**: A linear classifier cannot "fix" poor embeddings; it can only extract what is already linearly separable in the feature space.
- **Baseline Alignment**: This is the standard protocol for evaluating unsupervised graph embeddings (e.g., Node2Vec, DeepWalk), allowing for a fair comparison.
- **Simplicity**: Prevents the evaluation from becoming a hyperparameter tuning exercise for the classifier itself.

## Consequences
- **Lower Bound**: The reported accuracy represents a lower bound on the utility of the embeddings.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
