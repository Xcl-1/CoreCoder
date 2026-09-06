"""Credential-file boundaries for content-reading tools, including recursive grep."""

from pathlib import Path


def sensitive_path(path: Path) -> bool:
    def matches(candidate: Path) -> bool:
        parts = {part.casefold() for part in candidate.parts}
        name = candidate.name.casefold()
        if parts & {".ssh", ".aws", ".azure", ".kube", ".gnupg"}:
            return True
        if name in {".env.example", ".env.sample", ".env.template"}:
            return False
        return (
            name == ".env" or name.startswith(".env.")
            or name in {".netrc", "_netrc", ".npmrc", "credentials", "credentials.json",
                        "id_rsa", "id_ed25519", "id_ecdsa", "id_dsa"}
            or candidate.suffix.casefold() in {".pem", ".key", ".p12", ".pfx"}
        )
    return matches(path) or matches(path.expanduser().resolve())
