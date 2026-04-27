# ADR 0011: Weighted Edge List Format for C-Pipeline

**Status**: Accepted

## Context
The original LouvainNE implementation silently ignored edge weights if they were provided, or crashed if the format was unexpected. To support the Hybrid Graph, weights must be explicitly processed.

## Decision
Standardize on a `u v w` (source, target, weight) space-separated text format for all communication between the Python fusion layer and the C embedding pipeline.

## Rationale
- **Compatibility**: Ensures that the weight $w_{uv}$ (representing similarity confidence) actually influences the Louvain modularity optimization.
- **Simplicity**: Plain text files are easy to debug and portable across OS boundaries.

## Consequences
- **I/O Overhead**: Writing large weighted edge lists to disk is slower than binary formats, but acceptable given the embedding time.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
