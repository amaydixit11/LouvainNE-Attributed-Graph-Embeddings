# ADR 0009: Recursive Partitioning via Louvain (recpart)

**Status**: Accepted

## Context
Standard community detection produces a single "flat" partition of the graph. However, network embeddings benefit from capturing multi-scale structural information—from tiny local clusters to massive global communities.

## Decision
Implement **Recursive Partitioning** using the Louvain algorithm. The process follows these steps:
1. **Initial Partition**: Run Louvain on the original graph $\mathcal{G}_0$ to get partition $\mathcal{P}_1$.
2. **Graph Coarsening**: Collapse each community in $\mathcal{P}_1$ into a single "super-node." Edges between super-nodes are the sum of weights of edges between the constituent communities.
3. **Recursion**: Run Louvain on the coarsened graph $\mathcal{G}_1$ to get partition $\mathcal{P}_2$.
4. **Termination**: Repeat until the graph collapses into a single community or no further modularity improvement is possible.

The output is a **Hierarchy File** where each line corresponds to a node and lists its community ID at every level $\ell \in [0, L]$.

## Rationale
- **Multi-Resolution Analysis**: Captures the "nested" nature of networks. A node's identity is defined by its membership in a small group, which is part of a larger region, which is part of a global cluster.
- **Algorithmic Efficiency**: Louvain is $O(n \log n)$. Recursive collapsing reduces the number of nodes $n$ exponentially at each level, ensuring the total time is dominated by the first level.
- **Structure Preservation**: By aggregating weights during coarsening, the algorithm preserves the structural "skeleton" of the graph across scales.

## Rejected Alternatives
- **K-Means on Structural Features**: Rejected because it requires a pre-defined $k$ and doesn't naturally produce a hierarchy.
- **Girvan-Newman**: Rejected due to $O(n^3)$ complexity, making it infeasible for anything larger than a few hundred nodes.

## Consequences
- **Hierarchy Storage**: The output `hierarchy.txt` is larger than a flat partition file, though it remains linear in $N \times L$.
- **Aggregation Dependency**: The utility of this hierarchy depends entirely on the subsequent aggregation strategy (see ADR 0010).

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
