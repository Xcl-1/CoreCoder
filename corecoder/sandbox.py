"""Lightweight sandbox for tool execution.

Optional Docker-based isolation for bash commands and a path whitelist
for file writes.  Controlled by the ``CORECODER_SANDBOX`` env var.

Design principles:
- Zero additional dependencies (uses ``subprocess`` for Docker, same as BashTool)
- Docker is optional; when explicitly requested, unavailability fails closed
- Path whitelist blocks writes to system-sensitive directories
"""

import os
import shutil
from pathlib import Path

# ---- path whitelist -------------------------------------------------------

# directories that tools are NEVER allowed to write to
_POSIX_BLOCKED_DIRS = [
    "/etc",
    "/boot",
    "/System",
    "/Library/System",
]

# paths that are blocked regardless of OS
_BLOCKED_GLOBS = [
    "~/.ssh",
    "~/.gnupg",
    "~/.aws",
]


def _resolve_blocked() -> set[Path]:
    """Resolve blocked globs to absolute paths."""
    blocked: set[Path] = set()
    if os.name == "nt":
        candidates = [
            os.environ.get("SystemRoot"),
            os.environ.get("ProgramFiles"),
            os.environ.get("ProgramFiles(x86)"),
            os.environ.get("ProgramData"),
        ]
    else:
        candidates = _POSIX_BLOCKED_DIRS
    for d in candidates:
        if not d:
            continue
        p = Path(d)
        try:
            blocked.add(p.resolve())
        except OSError:
            # An unreadable sensitive path should never make package import fail.
            blocked.add(p.absolute())
    for g in _BLOCKED_GLOBS:
        p = Path(g).expanduser()
        try:
            blocked.add(p.resolve())
        except OSError:
            blocked.add(p.absolute())
    return blocked


# Resolved once at import; nonexistent credential directories are included too.
_BLOCKED_PATHS: set[Path] = _resolve_blocked()


def is_write_blocked(target: str | Path) -> bool:
    """Return True if writing to *target* should be blocked.

    Checks whether *target* is inside any system-sensitive directory.
    """
    try:
        resolved = Path(target).expanduser().resolve()
    except (OSError, RuntimeError):
        return True  # can't even resolve it — deny

    # Credential paths remain protected even if their parent directory does
    # not exist yet (otherwise creating ~/.ssh would bypass the import cache).
    from .tools.sensitive import sensitive_path
    if sensitive_path(Path(target)):
        return True

    for blocked in _BLOCKED_PATHS:
        try:
            resolved.relative_to(blocked)
            return True
        except ValueError:
            pass  # target is not under this blocked path
    return False


# ---- Docker sandbox -------------------------------------------------------

_DOCKER_AVAILABLE: bool | None = None  # None = unchecked


class SandboxUnavailableError(RuntimeError):
    """Raised when isolation was required but cannot be provided."""


def docker_available() -> bool:
    """Check whether Docker is installed and the daemon is reachable."""
    global _DOCKER_AVAILABLE
    if _DOCKER_AVAILABLE is None:
        import subprocess
        _DOCKER_AVAILABLE = (
            shutil.which("docker") is not None
            and subprocess.run(
                ["docker", "info"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode == 0
        )
    return _DOCKER_AVAILABLE


def sandbox_enabled() -> bool:
    """Return True if the user has opted into sandbox mode."""
    return os.getenv("CORECODER_SANDBOX", "").strip() in ("1", "true", "yes")


def wrap_command(command: str, cwd: str | None = None) -> str:
    """Wrap a shell command to run inside a Docker container.

    The container mounts the current working directory and runs as an
    unprivileged Alpine image.  Docker mode only applies when the user
    has set ``CORECODER_SANDBOX=1`` AND Docker is available.

    Raises ``SandboxUnavailableError`` when isolation was explicitly requested
    but Docker cannot provide it. Silent fallback would run an untrusted
    command on the host with more authority than the user selected.
    """
    if not sandbox_enabled():
        return command
    if not docker_available():
        raise SandboxUnavailableError("sandbox requested but Docker is unavailable")

    network_mode = os.getenv("CORECODER_SANDBOX_NETWORK", "none").strip().lower()
    if network_mode not in {"none", "bridge"}:
        raise SandboxUnavailableError(
            "CORECODER_SANDBOX_NETWORK must be 'none' or 'bridge'"
        )

    workdir = cwd or os.getcwd()
    # escape single quotes in the command for the sh -c wrapper
    escaped = command.replace("'", "'\\''")
    return (
        f"docker run --rm "
        f"{'--network none ' if network_mode == 'none' else ''}"
        f"-v '{workdir}':'{workdir}' "
        f"-w '{workdir}' "
        f"alpine:latest sh -c '{escaped}'"
    )
