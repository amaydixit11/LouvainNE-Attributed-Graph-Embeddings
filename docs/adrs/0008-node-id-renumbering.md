# ADR 0008: Node ID Renumbering (renum)

**Status**: Accepted

## Context
Real-world graph datasets (like Cora or BlogCatalog) often use non-contiguous, sparse, or string-based node identifiers. The core C-pipeline for Louvain partitioning and embedding generation relies on high-performance array indexing, where the index in the array corresponds directly to the node ID.

If node IDs are sparse (e.g., IDs 1, 100, 10000), allocating an array of size 10001 to store data for only 3 nodes is extremely wasteful and can lead to `OutOfMemory` errors or segmentation faults.

## Decision
Implement a dedicated `renum` (renumbering) stage that maps the original, potentially sparse set of IDs $\mathcal{V}_{orig}$ to a contiguous integer range $[0, N-1]$.

The mapping is handled as follows:
1. Scan the edge list to identify all unique node IDs.
2. Assign a unique integer from $0$ to $N-1$ to each unique ID in the order they appear or sorted.
3. Produce a new edge list using the renumbered IDs.
4. Export a `map.txt` file containing the "new\_label old\_label" pairs for result backtracking.

## Rationale
- **Memory Efficiency**: Allows the use of flat arrays (e.g., `unsigned long *node2Community`) instead of expensive hash maps or associative arrays in C.
- **CPU Cache Locality**: Contiguous indexing ensures that when the algorithm iterates over neighbors, it accesses memory in a more predictable pattern, reducing cache misses.
- **Simplicity**: Offloads the complexity of ID management to a pre-processing step, keeping the core Louvain implementation lean and fast.

## Rejected Alternatives
- **Using Hash Maps in C**: Rejected because `std::unordered_map` or custom C hash tables introduce significant overhead per access, slowing down the inner loop of the Louvain modularity optimization.
- **On-the-fly Mapping**: Rejected because it would require a lookup for every edge access, multiplying the number of memory accesses by the number of edges.

## Consequences
- **Preprocessing Overhead**: Adds a linear-time pass over the edge list before the main pipeline.
- **Mapping Maintenance**: Requires the `hi2vec` and `renum` tools to strictly track the mapping file to ensure that the final embeddings are assigned to the correct original nodes.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
