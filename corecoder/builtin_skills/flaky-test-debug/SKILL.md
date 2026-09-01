# Goal

Replace nondeterministic test behavior with a deterministic explanation and correction.

# Guidance

- Re-run under controlled seeds, order, time, concurrency, and environment while preserving failure evidence.
- Inspect shared state, clocks, randomness, cleanup, ports, filesystem, and external services.
- Fix the underlying isolation or synchronization issue; do not mask it with retries or generous sleeps.

