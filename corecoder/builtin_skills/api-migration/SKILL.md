# Goal

Move API producers and consumers to the target contract without an accidental breaking change.

# Guidance

- Inventory callers, schemas, generated clients, tests, and documented compatibility promises.
- Prefer an additive transition with adapters or deprecation periods when both versions must coexist.
- Verify old and new paths, then remove compatibility code only when explicitly in scope.

