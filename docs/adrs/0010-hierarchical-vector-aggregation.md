# ADR 0010: Hierarchical Vector Aggregation (hi2vec)

**Status**: Accepted

## Context
We have a community hierarchy (from `recpart`) that assigns each node $v$ to a sequence of communities $c_0, c_1, \dots, c_L$. We need to map this discrete symbolic path into a continuous $D$-dimensional embedding vector $\mathbf{z}_v$.

## Decision
Implement **Weighted Random Vector Aggregation**:
1. **Random Assignment**: For every community $C_{i,\ell}$ at every level $\ell$, assign a random vector $\mathbf{r}_{i,\ell} \in \mathbb{R}^D$ sampled from a uniform distribution $[-1, 1]$.
2. **Weighted Sum**: The final embedding is the normalized sum:
   $$\mathbf{z}_v = \text{Normalize}\left(\sum_{\ell=0}^L w_\ell \mathbf{r}_{c_\ell(v)}\right)$$
3. **Exponential Decay**: Use weights $w_\ell = e^{-\lambda \ell}$, where $\lambda$ is a damping factor.

## Rationale
- **Training-Free Representation**: By using random vectors, we avoid the need for an expensive training phase (like Word2Vec) while still ensuring that nodes in the same community share the same embedding components.
- **Curse of Dimensionality**: In high-dimensional space ($D=256$), random vectors are naturally nearly orthogonal. This ensures that different communities are represented by distinct, non-overlapping signals.
- **Tuning Granularity**: The $\lambda$ parameter allows us to explicitly control the "zoom" of the embedding: a high $\lambda$ emphasizes local structure, while a low $\lambda$ emphasizes global structure.

## Rejected Alternatives
- **One-Hot Encoding**: Rejected because it would lead to extremely high-dimensional, sparse vectors that are unsuitable for linear probes.
- **Learned Community Embeddings**: Rejected because it would introduce a training phase and potential overfitting to the training split.

## Consequences
- **Seed Sensitivity**: Results depend on the random seed used for $\mathbf{r}_{i,\ell}$. This requires strict seed management (ADR 0012) to ensure reproducibility.
- **Lossy Compression**: The embedding is a "summary" of the hierarchy; a specific node's unique identity is lost if it is the only member of its community at all levels.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
