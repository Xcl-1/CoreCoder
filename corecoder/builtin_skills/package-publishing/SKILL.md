# Goal

Produce a validated package artifact and publish it only to the explicitly authorized registry and target.

# Guidance

- Confirm package metadata, version uniqueness, credentials mechanism, registry, and release channel.
- Build clean artifacts and inspect their contents before any upload.
- Treat publishing, tagging, and pushing as external mutations requiring explicit scope; stop after validation when authorization is absent.

