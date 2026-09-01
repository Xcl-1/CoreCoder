# Goal

Produce a reproducible container image that runs the application with minimal attack surface and build waste.

# Guidance

- Determine build inputs, runtime command, ports, writable paths, health behavior, and target platform.
- Use deterministic dependencies, effective build caching, a non-root runtime where practical, and no embedded secrets.
- Verify the image builds and the documented invocation starts the intended service.

