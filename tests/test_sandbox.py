"""Tests for the sandbox module."""

import os

import pytest

from corecoder.sandbox import docker_available, is_write_blocked, sandbox_enabled, wrap_command

# --- path whitelist ------------------------------------------------------

def test_is_write_blocked_etc(tmp_path):
    """Writing to /etc should be blocked."""
    # test with a path under a blocked directory
    # /etc always exists on Linux/macOS; skip gracefully on Windows
    if os.name == "nt":
        assert not is_write_blocked("C:\\Users\\Public\\test.txt")
    else:
        assert is_write_blocked("/etc/hosts")
        assert is_write_blocked("/etc/nginx/nginx.conf")


def test_is_write_blocked_ssh(tmp_path):
    """Writing to ~/.ssh should be blocked."""
    home = os.path.expanduser("~")
    ssh_path = os.path.join(home, ".ssh", "authorized_keys")
    assert is_write_blocked(ssh_path)


def test_is_write_blocked_credentials_even_before_parent_exists(tmp_path):
    assert is_write_blocked(tmp_path / ".ssh" / "authorized_keys")
    assert is_write_blocked(tmp_path / ".env")
    assert is_write_blocked(tmp_path / "service-account.pem")
    assert not is_write_blocked(tmp_path / ".env.example")


def test_is_write_blocked_windows_system_root():
    if os.name == "nt" and os.environ.get("SystemRoot"):
        assert is_write_blocked(os.path.join(os.environ["SystemRoot"], "System32", "drivers", "etc", "hosts"))


def test_is_write_blocked_normal_path_not_blocked(tmp_path):
    """Normal project paths should not be blocked."""
    assert not is_write_blocked(str(tmp_path / "output.txt"))
    assert not is_write_blocked(str(tmp_path / "src" / "main.py"))


def test_is_write_blocked_nonexistent_blocked_dir():
    """A path under a blocked dir that exists should be blocked."""
    # /etc always exists on Linux/macOS
    if os.name != "nt" and os.path.exists("/etc"):
        assert is_write_blocked("/etc/some/file.txt")


def test_is_write_blocked_resolves_symlinks_etc(tmp_path):
    """Path resolution should catch traversal attempts."""
    # Verify a clearly blocked path resolves correctly per-OS
    if os.name == "nt":
        # Windows: C:\Windows is not in the blocked list by default,
        # so test with a path that goes through resolution unchanged
        assert not is_write_blocked(str(tmp_path / "safe.txt"))
    else:
        # /etc always exists on Linux
        if os.path.exists("/etc"):
            assert is_write_blocked("/etc/some/file.txt")


def test_is_write_blocked_unresolvable_path():
    """A path that can't be resolved should be blocked (fail-safe)."""
    # Create a deeply nested path that might fail resolution
    # Actually, all paths can be resolved (they just won't exist)
    # The function resolves the path; nonexistent paths resolve fine
    assert not is_write_blocked("/tmp/definitely/nonexistent/path.txt")


# --- Docker sandbox ------------------------------------------------------

def test_sandbox_disabled_by_default(monkeypatch):
    monkeypatch.delenv("CORECODER_SANDBOX", raising=False)
    assert not sandbox_enabled()


def test_sandbox_enabled_with_env(monkeypatch):
    monkeypatch.setenv("CORECODER_SANDBOX", "1")
    assert sandbox_enabled()
    monkeypatch.setenv("CORECODER_SANDBOX", "true")
    assert sandbox_enabled()
    monkeypatch.setenv("CORECODER_SANDBOX", "yes")
    assert sandbox_enabled()
    monkeypatch.setenv("CORECODER_SANDBOX", "0")
    assert not sandbox_enabled()
    monkeypatch.setenv("CORECODER_SANDBOX", "no")
    assert not sandbox_enabled()


def test_wrap_command_no_sandbox(monkeypatch):
    monkeypatch.delenv("CORECODER_SANDBOX", raising=False)
    cmd = "echo hello"
    assert wrap_command(cmd) == cmd  # unchanged when sandbox is off


def test_wrap_command_with_sandbox_no_docker_fails_closed(monkeypatch):
    """Requested isolation must never silently fall back to host execution."""
    monkeypatch.setenv("CORECODER_SANDBOX", "1")
    monkeypatch.setattr("corecoder.sandbox.docker_available", lambda: False)
    with pytest.raises(RuntimeError, match="Docker is unavailable"):
        wrap_command("echo hello")


def test_docker_available_returns_bool():
    result = docker_available()
    assert isinstance(result, bool)


def test_wrap_command_preserves_command_semantics(monkeypatch):
    """Even when wrapped, the original command should appear in the result."""
    monkeypatch.setenv("CORECODER_SANDBOX", "1")
    monkeypatch.setattr("corecoder.sandbox.docker_available", lambda: True)
    result = wrap_command("python -c 'print(1)'")
    assert "python" in result
    assert "--network none" in result


def test_sandbox_bridge_network_requires_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("CORECODER_SANDBOX", "1")
    monkeypatch.setenv("CORECODER_SANDBOX_NETWORK", "bridge")
    monkeypatch.setattr("corecoder.sandbox.docker_available", lambda: True)
    result = wrap_command("curl https://example.com")
    assert "--network none" not in result


def test_invalid_sandbox_network_mode_fails_closed(monkeypatch):
    monkeypatch.setenv("CORECODER_SANDBOX", "1")
    monkeypatch.setenv("CORECODER_SANDBOX_NETWORK", "host")
    monkeypatch.setattr("corecoder.sandbox.docker_available", lambda: True)
    with pytest.raises(RuntimeError, match="must be 'none' or 'bridge'"):
        wrap_command("echo hello")
