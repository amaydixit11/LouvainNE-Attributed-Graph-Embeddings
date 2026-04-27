# ADR 0031: L-BFGS Optimizer for Linear Probing

**Status**: Accepted

## Context
The linear probe (Logistic Regression) requires an optimization algorithm to find the weights that maximize the micro-F1 score on the training set.

## Decision
Use the **L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno)** optimizer with strong-Wolfe line search.

## Rationale
- **Convergence**: L-BFGS converges significantly faster than standard Gradient Descent for small-to-medium sized linear problems.
- **Memory Efficiency**: It approximates the Hessian matrix using a limited amount of memory, making it suitable for high-dimensional embedding spaces.
- **Stability**: The strong-Wolfe line search ensures stable step sizes, reducing the need for manual learning rate tuning.

## Consequences
- **Deterministic Results**: L-BFGS provides highly consistent results across different runs on the same data.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
