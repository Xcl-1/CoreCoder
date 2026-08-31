# Goal

Improve structure or naming without changing observable behavior.

# Workflow

1. Map the target's callers, tests, imports, and public API surface.
2. State the behavior that must remain invariant.
3. Prefer a sequence of small, mechanically verifiable edits.
4. Update all references and preserve compatibility where required.
5. Run focused tests after each risky boundary, followed by the relevant suite.

# Boundaries

- Keep feature changes and bug fixes out of a pure refactor.
- Do not rename a public symbol without checking downstream compatibility.
- Avoid broad rewrites when a targeted transformation is sufficient.

# Validation

Confirm that references are consistent, syntax remains valid, and observable behavior is covered by tests.
