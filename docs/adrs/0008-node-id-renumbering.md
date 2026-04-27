# ADR 0008: Node ID Renumbering (renum)

**Status**: Accepted

## Context
Input graphs often have sparse or non-contiguous node IDs. The core C-pipeline for community detection requires node IDs to be in the range $[0, N-1]$ for efficient array-based indexing.

## Decision
Implement a `renum` stage that maps original IDs to a contiguous integer range while preserving the original-to-new mapping in a metadata file.

## Rationale
- **Memory Efficiency**: Allows the use of flat arrays instead of hash maps for community assignments and edge lists in C.
- **Performance**: Direct index access is significantly faster than lookups.

## Consequences
- **Mapping Overhead**: Requires a translation step when mapping results back to the original dataset IDs.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
