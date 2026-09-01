# Goal

Change persisted schemas or data without corrupting records or unexpectedly breaking mixed-version deployments.

# Guidance

- Inventory readers, writers, constraints, volume, locking risk, and rollback limitations.
- Prefer expand-migrate-contract sequencing when old and new application versions can overlap.
- Validate forward behavior, representative data, and recovery strategy before destructive cleanup.

