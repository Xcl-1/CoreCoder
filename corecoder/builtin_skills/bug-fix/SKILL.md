# Goal

Establish the root cause of incorrect behavior and make the smallest justified fix.

# Workflow

1. Reproduce the failure or derive a precise triggering condition from available evidence.
2. Trace the failing path through callers, state changes, and boundary conditions.
3. Distinguish the root cause from downstream symptoms.
4. Implement a focused correction and add a regression test when practical.
5. Re-run the reproduction and relevant surrounding tests.

# Boundaries

- Do not edit before there is evidence for the failure mechanism.
- Avoid broad cleanup unless it is necessary for the correction.
- State clearly when the original failure could not be reproduced.

