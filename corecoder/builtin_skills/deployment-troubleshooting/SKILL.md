# Goal

Locate the first failing deployment boundary and restore expected runtime behavior without unsafe production assumptions.

# Guidance

- Separate build, scheduling, startup, configuration, dependency, health, networking, and traffic-routing stages.
- Correlate manifests and code with available logs; never invent access to live infrastructure.
- Prefer reversible corrections and state when production verification or rollback requires user authority.

