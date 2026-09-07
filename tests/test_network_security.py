"""Network intent, SSRF, allowlist, confirmation, and audit tests."""

import json

import pytest

from corecoder.security import (
    AuditLogger,
    Guard,
    NetworkPolicy,
    PermissionManager,
    PermissionRule,
    inspect_network_intent,
)
from corecoder.security.capabilities import inspect_tool_capabilities
from corecoder.tools import get_tool


@pytest.fixture(autouse=True)
def _isolate_security_state(tmp_path, monkeypatch):
    monkeypatch.setattr("corecoder.security.audit.AUDIT_DIR", tmp_path / "audit")
    monkeypatch.setattr(
        "corecoder.security.permissions.USER_PERMISSIONS_PATH",
        tmp_path / "user-permissions.json",
    )
    monkeypatch.setattr(
        "corecoder.security.permissions.PROJECT_PERMISSIONS_PATH",
        tmp_path / "project-permissions.json",
    )


def _allow_bash() -> PermissionManager:
    manager = PermissionManager()
    manager._user_rules = [PermissionRule("bash", ".*", "allow", "test", 1000, "user")]
    manager._all_sorted = None
    return manager


def test_curl_intent_extracts_destination_and_mutation():
    intent = inspect_network_intent(
        "bash",
        {"command": "curl -X POST --data payload https://api.example.com/v1/items"},
    )
    assert intent.accesses_network is True
    assert intent.destinations == ("api.example.com",)
    assert intent.mutating is True


@pytest.mark.parametrize(
    "command",
    [
        "git -C repo push origin main",
        "python -m pip install requests",
        "uv pip install httpx",
        "ssh deploy@example.com restart-service",
        "scp build.zip deploy@example.com:/srv/releases/",
        "dig secret-data.example.com",
        "nc example.com 443",
        "go get example.com/module",
    ],
)
def test_common_non_http_egress_commands_are_detected(command):
    assert inspect_network_intent("bash", {"command": command}).accesses_network is True


def test_ssh_policy_uses_first_positional_host_not_remote_command_argument():
    intent = inspect_network_intent("bash", {"command": "ssh evil.example.com approved.example.com"})
    assert intent.destinations == ("evil.example.com",)


def test_git_ssh_url_destination_is_extracted():
    intent = inspect_network_intent("bash", {"command": "git clone ssh://git@example.com/org/repo"})
    assert intent.destinations == ("example.com",)


@pytest.mark.parametrize("command", ["curl http://[::1", "curl http://[broken", "curl http://%zz"])
def test_malformed_network_targets_fail_to_confirmation_not_exception(command):
    decision = NetworkPolicy().review("bash", {"command": command})
    assert decision.action == "ask"


@pytest.mark.parametrize(
    "command",
    [
        "git -C repo push origin main",
        "ssh deploy@example.com restart-service",
        "scp build.zip deploy@example.com:/srv/releases/",
        "wget --post-data payload https://example.com/hook",
        "curl -T artifact.zip https://example.com/upload",
    ],
)
def test_remote_mutation_variants_are_classified(command):
    assert inspect_network_intent("bash", {"command": command}).mutating is True


@pytest.mark.parametrize("command", ["git status", "echo https://example.com", "python --version"])
def test_non_network_commands_do_not_trigger_egress_policy(command):
    assert inspect_network_intent("bash", {"command": command}).accesses_network is False


def test_opaque_interpreter_is_network_capable_at_execution_boundary():
    tool = get_tool("bash")
    arguments = {"command": "python opaque_probe.py"}
    capability = inspect_tool_capabilities(tool, arguments)

    intent = inspect_network_intent("bash", arguments, capability)

    assert intent.accesses_network is True
    assert intent.destinations == ()
    assert intent.client == "opaque-process"


def test_known_read_only_command_remains_offline_at_execution_boundary():
    tool = get_tool("bash")
    arguments = {"command": "git status"}
    capability = inspect_tool_capabilities(tool, arguments)

    assert inspect_network_intent("bash", arguments, capability).accesses_network is False


def test_allowlisted_read_only_destination_is_allowed():
    policy = NetworkPolicy(allowed_hosts=("api.example.com",))
    decision = policy.review("bash", {"command": "curl https://api.example.com/data"})
    assert decision.action == "allow"


def test_wildcard_allowlist_matches_subdomains_not_apex():
    policy = NetworkPolicy(allowed_hosts=("*.example.com",))
    assert policy.review("bash", {"command": "curl https://api.example.com"}).action == "allow"
    assert policy.review("bash", {"command": "curl https://example.com"}).action == "ask"


@pytest.mark.parametrize(
    "host",
    [
        "169.254.169.254",
        "2852039166",
        "0xA9FEA9FE",
        "[fd00:ec2::254]",
        "metadata.google.internal",
        "100.100.100.200",
    ],
)
def test_cloud_metadata_and_link_local_targets_are_denied(host):
    decision = NetworkPolicy().review("bash", {"command": f"curl http://{host}/metadata"})
    assert decision.action == "deny"
    assert "metadata or link-local" in decision.reason


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "10.0.0.8", "[::1]"])
def test_private_and_loopback_targets_require_confirmation(host):
    decision = NetworkPolicy().review("bash", {"command": f"curl http://{host}/admin"})
    assert decision.action == "ask"
    assert "private or loopback" in decision.reason


def test_url_embedded_credentials_are_denied():
    decision = NetworkPolicy(allowed_hosts=("example.com",)).review(
        "bash", {"command": "curl https://alice:secret@example.com/private"}
    )
    assert decision.action == "deny"
    assert "embedded credentials" in decision.reason


@pytest.mark.parametrize(
    "command",
    [
        "curl https://example.com/data?access_token=secret",
        "curl https://example.com/data?api_key=secret",
    ],
)
def test_credentials_in_url_query_are_denied(command):
    assert NetworkPolicy(allowed_hosts=("example.com",)).review("bash", {"command": command}).action == "deny"


@pytest.mark.parametrize(
    "command",
    [
        'curl -H "Authorization: Bearer secret" https://example.com/data',
        "curl -u alice:secret https://example.com/data",
    ],
)
def test_authentication_material_requires_confirmation(command):
    decision = NetworkPolicy(allowed_hosts=("example.com",)).review("bash", {"command": command})
    assert decision.action == "ask"
    assert "authentication material" in decision.reason


@pytest.mark.parametrize(
    "command",
    [
        "curl -L https://example.com/data",
        "curl --location https://example.com/data",
        "wget https://example.com/data",
    ],
)
def test_redirect_following_requires_confirmation_even_when_allowlisted(command):
    decision = NetworkPolicy(allowed_hosts=("example.com",)).review("bash", {"command": command})
    assert decision.action == "ask"
    assert "redirect" in decision.reason


def test_mutating_network_call_requires_confirmation_even_when_allowlisted():
    decision = NetworkPolicy(allowed_hosts=("api.example.com",)).review(
        "bash", {"command": "curl -d payload https://api.example.com/items"}
    )
    assert decision.action == "ask"
    assert "mutate remote state" in decision.reason


def test_network_deny_mode_blocks_unknown_destination():
    policy = NetworkPolicy(mode="deny", allowed_hosts=("approved.example.com",))
    assert policy.review("bash", {"command": "curl https://other.example.com"}).action == "deny"
    assert policy.review("bash", {"command": "pip install requests"}).action == "deny"


@pytest.mark.parametrize(
    "entry",
    ["https://example.com", "user@example.com", "*.com", "127.0.0.1", "metadata.google.internal"],
)
def test_invalid_or_overbroad_allowlist_entries_are_rejected(entry):
    with pytest.raises(ValueError):
        NetworkPolicy(allowed_hosts=(entry,))


def test_guard_network_confirmation_cannot_be_bypassed_by_permission_allow():
    guard = Guard(
        permissions=_allow_bash(),
        network_policy=NetworkPolicy(allowed_hosts=("approved.example.com",)),
    )
    decision = guard.review(
        "bash",
        {"command": "curl https://unapproved.example.com/data"},
        tool=get_tool("bash"),
    )
    assert decision.allowed is False
    assert decision.network.action == "ask"
    assert "not allowlisted" in decision.reason


def test_permission_allow_cannot_bypass_opaque_process_network_review():
    guard = Guard(permissions=_allow_bash())

    decision = guard.review(
        "bash",
        {"command": "python opaque_probe.py"},
        tool=get_tool("bash"),
    )

    assert decision.allowed is False
    assert decision.network.intent.client == "opaque-process"
    assert "no statically verifiable destination" in decision.reason


def test_guard_allows_read_only_allowlisted_call_with_permission():
    guard = Guard(
        permissions=_allow_bash(),
        network_policy=NetworkPolicy(allowed_hosts=("approved.example.com",)),
    )
    decision = guard.review(
        "bash",
        {"command": "curl https://approved.example.com/data"},
        tool=get_tool("bash"),
    )
    assert decision.allowed is True
    assert decision.network.action == "allow"


def test_guard_elevates_remote_mutation_to_high_risk():
    decision = Guard(
        permissions=_allow_bash(),
        confirm_callback=lambda *_args: True,
        network_policy=NetworkPolicy(allowed_hosts=("api.example.com",)),
    ).review(
        "bash",
        {"command": "curl -T artifact.zip https://api.example.com/upload"},
        tool=get_tool("bash"),
    )
    assert decision.allowed is True
    assert decision.risk.level.name == "HIGH"


def test_metadata_deny_wins_even_with_permission_and_confirmation():
    guard = Guard(
        permissions=_allow_bash(),
        confirm_callback=lambda *_args: True,
    )
    decision = guard.review(
        "bash",
        {"command": "curl http://169.254.169.254/latest/meta-data"},
        tool=get_tool("bash"),
    )
    assert decision.allowed is False
    assert decision.network.action == "deny"


def test_network_audit_records_policy_and_destinations(tmp_path):
    audit_dir = tmp_path / "network-audit"
    guard = Guard(
        permissions=_allow_bash(),
        audit=AuditLogger(audit_dir),
        network_policy=NetworkPolicy(allowed_hosts=("approved.example.com",)),
    )
    guard.review(
        "bash",
        {"command": "curl https://approved.example.com/data"},
        tool=get_tool("bash"),
    )
    payload = json.loads(next(audit_dir.glob("audit_*.jsonl")).read_text(encoding="utf-8"))
    assert payload["network_policy_action"] == "allow"
    assert payload["network_destinations"] == ["approved.example.com"]
