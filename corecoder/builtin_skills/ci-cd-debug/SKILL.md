# Goal

Explain and correct the pipeline failure using the smallest reproducible difference from local execution.

# Guidance

- Locate the exact failing job and compare runtime, environment, permissions, cache, paths, and artifacts.
- Reproduce the failing command locally when safe and inspect the earliest meaningful error.
- Avoid disabling checks or broadening permissions merely to make the pipeline green.

