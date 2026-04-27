# ADR 0013: Community Collapse Safety Patch

**Status**: Accepted

## Context
In some datasets, the Louvain algorithm can collapse the entire graph into a single community in a single step. This caused a null-pointer dereference in `recpart.c` when it attempted to iterate over the (now non-existent) second level of the hierarchy.

## Decision
Add explicit checks in `recpart.c` to detect when the number of communities reaches 1, and gracefully terminate the recursive partitioning.

## Rationale
- **Robustness**: Prevents segmentation faults on "trivial" or highly connected graphs.
- **Correctness**: A single-community graph is a valid result and should be handled without crashing.

## Consequences
- **Edge Case Handling**: Adds a small amount of conditional logic to the inner loop of the C partitioner.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
