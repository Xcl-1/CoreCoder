# Goal

Explain the concurrency failure as an ordering or ownership violation and correct it safely.

# Guidance

- Map tasks, threads, shared state, synchronization, cancellation, and lifecycle boundaries.
- Seek a deterministic reproduction or stress harness without treating timing sleeps as a fix.
- Verify safety properties as well as the original symptom after the correction.

