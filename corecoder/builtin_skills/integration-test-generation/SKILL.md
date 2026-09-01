# Goal

Verify the real contract between components without creating brittle end-to-end coverage.

# Guidance

- Identify the boundary, ownership of fixtures, and which dependencies must be real versus controlled.
- Test transactions, serialization, cleanup, failure propagation, and one representative happy path.
- Keep tests isolated and deterministic, and document any required external service.

