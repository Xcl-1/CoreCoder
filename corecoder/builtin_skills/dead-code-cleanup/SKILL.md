# Goal

Reduce maintenance surface by removing code that is demonstrably unreachable or unused.

# Guidance

- Verify references across imports, dynamic registration, configuration, packaging, tests, and public APIs.
- Remove the smallest coherent unit and update affected tests or documentation.
- Run regression checks and call out uncertainty around reflective or external consumers.

