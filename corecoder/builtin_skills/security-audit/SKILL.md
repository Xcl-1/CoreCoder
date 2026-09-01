# Goal

Identify concrete security weaknesses and explain their trigger, impact, and remediation direction without changing files.

# Workflow

1. Map trust boundaries, entry points, identities, privileges, and sensitive data flows.
2. Inspect validation, authorization, command or query construction, path handling, and secret exposure.
3. Trace each suspected issue to a reachable condition and verify existing safeguards.
4. Rank confirmed findings by exploitability and impact with precise file evidence.

# Boundaries

- Remain read-only and do not execute potentially harmful proof-of-concept actions.
- Do not report theoretical issues that are blocked by verified controls.
- Clearly distinguish confirmed findings from verification gaps.

