# Goal

Make failure behavior predictable, diagnosable, and safe without hiding actionable errors.

# Guidance

- Classify expected, transient, permanent, and programmer failures at system boundaries.
- Preserve causes, guarantee cleanup, and retry only idempotent transient operations with bounds.
- Test failure paths and ensure diagnostics contain context without leaking secrets.

