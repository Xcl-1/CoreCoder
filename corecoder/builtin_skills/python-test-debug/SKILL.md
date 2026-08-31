# Goal

Find the root cause of a Python test failure, make the smallest justified correction when requested, and verify the fix.

# Workflow

1. Run the narrowest failing test and preserve the full error signal.
2. Read the failing test, implementation, and directly relevant fixtures or callers.
3. Separate product defects, test defects, environment problems, and flaky behavior.
4. Apply the smallest change consistent with intended behavior.
5. Re-run the narrow test, then the relevant wider suite.

# Boundaries

- Never weaken or delete an assertion merely to make the suite green.
- Do not edit before reproducing or otherwise establishing the failure cause.
- Keep dependency and environment changes separate from code fixes.

# Validation

Report the root cause, changed behavior, narrow verification, and any broader suite result.
