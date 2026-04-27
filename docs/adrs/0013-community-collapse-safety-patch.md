# ADR 0013: Community Collapse Safety Patch

**Status**: Accepted

## Context
The recursive partitioning process in `recpart.c` assumes that the graph will continue to be split into multiple communities at each level. However, on certain "trivial" graphs (e.g., very small graphs or extremely dense cliques), the Louvain algorithm can collapse the entire graph into a single community in a single step.

This caused a **null-pointer dereference** because the code attempted to iterate over the children of the root community, but the collapse had already terminated the hierarchy.

## Decision
Implement an **explicit collapse check** in the recursive loop of `recpart.c`. If the number of communities $n_{lab}$ produced by a partition is 1, the recursion terminates immediately and the current level is marked as the leaf.

## Rationale
- **Robustness**: Prevents segmentation faults on edge-case datasets.
- **Correctness**: A graph that cannot be further partitioned is a valid result and should be handled gracefully rather than treated as an error.

## Rejected Alternatives
- **Adding Dummy Edges**: Rejected because it would distort the modularity and produce a fake hierarchy.
- **Increasing Memory Limits**: Rejected because the issue was a logic error (null pointer), not a resource limit.

## Consequences
- **Implementation Detail**: Adds a simple `if (nlab == 1)` check in the `recurs` function of the C code.
- **Hierarchy Depth**: Some graphs will have a much shallower hierarchy than others, which is handled naturally by the `hi2vec` aggregation logic.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
