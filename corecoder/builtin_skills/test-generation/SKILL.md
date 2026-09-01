# Goal

Create maintainable tests that demonstrate behavior rather than implementation details.

# Workflow

1. Read the target behavior, existing tests, fixtures, and project test conventions.
2. Identify representative success, failure, and boundary cases.
3. Add the smallest test set that provides distinct behavioral coverage.
4. Run the new tests and relevant existing tests.

# Boundaries

- Prefer public interfaces and observable outcomes over private implementation details.
- Avoid redundant cases that exercise the same path without adding confidence.
- Do not weaken assertions merely to make a test pass.

