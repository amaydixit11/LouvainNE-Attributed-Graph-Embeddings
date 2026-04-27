# ADR 0019: Incremental Results Persistence

**Status**: Accepted

## Context
Large-scale benchmarks (e.g., the 1M node graph) can run for hours. A crash at the end of a 10-hour run results in total data loss.

## Decision
Implement **incremental JSON writes**: save the current results list to disk immediately after each individual graph size is processed.

## Rationale
- **Fault Tolerance**: Ensures that partial results are preserved if the process is killed by OOM or system failure.
- **Observability**: Allows the user to monitor progress by checking the JSON file in real-time.

## Consequences
- **Disk I/O**: Slight increase in writes, but negligible compared to the embedding time.

---
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
