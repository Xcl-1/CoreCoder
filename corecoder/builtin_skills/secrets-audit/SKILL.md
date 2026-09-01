# Goal

Identify concrete credential exposure paths without reproducing or revealing secret values.

# Guidance

- Inspect tracked configuration, examples, logs, test fixtures, command construction, and secret-loading boundaries.
- Redact values in findings and distinguish placeholders from active-looking credentials.
- Explain rotation and remediation needs without attempting external credential use.

