"""Local-only live smoke tests for the security execution boundary.

Run explicitly from the repository root::

    python smoke_tests/test_security_live.py

The test opens an ephemeral HTTP server bound only to 127.0.0.1.  It never
contacts the public internet and does not require API credentials.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from corecoder.agent import Agent
from corecoder.models import ToolCall
from corecoder.sandbox import docker_available
from corecoder.security import AuditLogger, Guard, PermissionManager, PermissionRule
from corecoder.tools.bash import BashTool


class _TargetHandler(BaseHTTPRequestHandler):
    events: list[dict[str, str]]

    def _record(self, body: bytes = b"") -> None:
        self.events.append(
            {
                "method": self.command,
                "path": self.path,
                "body": body.decode("utf-8", errors="replace"),
            }
        )

    def do_GET(self) -> None:
        self._record()
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/final")
            self.end_headers()
            return
        payload = b"live-final" if self.path == "/final" else b"live-ok"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self._record(body)
        payload = b"posted"
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *_args: object) -> None:
        return


@contextmanager
def _local_target():
    events: list[dict[str, str]] = []

    class Handler(_TargetHandler):
        pass

    Handler.events = events
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", events
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _permissions() -> PermissionManager:
    manager = PermissionManager()
    manager._user_rules = []
    manager._project_rules = []
    manager.add_session_rule(
        PermissionRule("bash", r".*", "allow", "live smoke test", 10_000, "session")
    )
    return manager


def _agent(audit_dir: Path, confirmation: bool | None) -> tuple[Agent, list[str]]:
    prompts: list[str] = []

    def confirm(_tool: str, _arguments: dict, reason: str) -> bool | None:
        prompts.append(reason)
        return confirmation

    guard = Guard(
        permissions=_permissions(),
        audit=AuditLogger(audit_dir),
        confirm_callback=None if confirmation is None else confirm,
    )
    agent = Agent(
        llm=object(),
        tools=[BashTool()],
        replay=False,
        guard=guard,
        context_artifacts_enabled=False,
    )
    return agent, prompts


async def _run(agent: Agent, command: str, call_id: str) -> tuple[str, bool]:
    result, _elapsed, success = await agent._exec_tool(
        ToolCall(id=call_id, name="bash", arguments={"command": command, "timeout": 15})
    )
    return result, success


def _audit_rows(audit_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in audit_dir.glob("audit_*.jsonl"):
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    return rows


async def _exercise_network(tmp: Path) -> dict[str, str]:
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl is None:
        raise AssertionError("curl is required for the live network smoke test")

    audit_dir = tmp / "audit"
    with _local_target() as (base_url, events):
        denied_agent, _ = _agent(audit_dir, None)
        before = len(events)
        result, success = await _run(
            denied_agent,
            f'"{curl}" --silent --show-error {base_url}/ok',
            "denied-get",
        )
        assert not success and "requires confirmation" in result
        assert len(events) == before, "a denied request reached the local target"

        confirmed_agent, prompts = _agent(audit_dir, True)
        result, success = await _run(
            confirmed_agent,
            f'"{curl}" --silent --show-error {base_url}/ok',
            "confirmed-get",
        )
        assert success and "live-ok" in result
        assert any("private or loopback" in reason for reason in prompts)

        result, success = await _run(
            confirmed_agent,
            f'"{curl}" --silent --show-error -L {base_url}/redirect',
            "confirmed-redirect",
        )
        assert success and "live-final" in result

        result, success = await _run(
            confirmed_agent,
            f'"{curl}" --silent --show-error -X POST --data live-payload {base_url}/upload',
            "confirmed-post",
        )
        assert success and "posted" in result
        assert any(event == {"method": "POST", "path": "/upload", "body": "live-payload"}
                   for event in events)

        before = len(events)
        result, success = await _run(
            confirmed_agent,
            f'"{curl}" --silent --max-time 1 http://169.254.169.254/latest/meta-data',
            "metadata",
        )
        assert not success and "metadata or link-local" in result
        assert len(events) == before

        # An arbitrary interpreter can hide network activity inside a script.
        # This must still require approval even though its command line does not
        # name urllib/requests/curl.
        probe = tmp / "opaque_network_probe.py"
        probe.write_text(
            "import sys\nimport urllib.request\nprint(urllib.request.urlopen(sys.argv[1]).read().decode())\n",
            encoding="utf-8",
        )
        before = len(events)
        result, success = await _run(
            denied_agent,
            f'"{sys.executable}" "{probe}" "{base_url}/opaque"',
            "opaque-interpreter",
        )
        assert not success and "requires confirmation" in result
        assert len(events) == before, "opaque interpreter bypassed network review"

    rows = _audit_rows(audit_dir)
    assert any(row["decision"] == "allow" and row["user_confirmed"] for row in rows)
    assert any(row["network_policy_action"] == "deny" for row in rows)
    assert any("127.0.0.1" in row["network_destinations"] for row in rows)
    return {
        "network": "denial, confirmation, redirect, POST, SSRF, opaque interpreter",
        "audit": f"{len(rows)} records verified",
    }


async def _exercise_shells(tmp: Path) -> dict[str, str]:
    audit_dir = tmp / "shell-audit"
    agent, _ = _agent(audit_dir, True)
    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    cmd = shutil.which("cmd.exe") or shutil.which("cmd")
    assert powershell is not None and cmd is not None

    ps_result, ps_success = await _run(
        agent,
        f'"{powershell}" -NoProfile -NonInteractive -Command "Write-Output live-powershell"',
        "powershell",
    )
    cmd_result, cmd_success = await _run(
        agent,
        f'"{cmd}" /d /c echo live-cmd',
        "cmd",
    )
    assert ps_success and "live-powershell" in ps_result
    assert cmd_success and "live-cmd" in cmd_result
    return {"powershell": "executed", "cmd": "executed"}


async def _exercise_sandbox_failure() -> dict[str, str]:
    if docker_available():
        return {"docker": "available; missing-runtime failure test skipped"}
    previous = os.environ.get("CORECODER_SANDBOX")
    os.environ["CORECODER_SANDBOX"] = "1"
    try:
        result = await BashTool().execute(command="echo must-not-run", timeout=10)
    finally:
        if previous is None:
            os.environ.pop("CORECODER_SANDBOX", None)
        else:
            os.environ["CORECODER_SANDBOX"] = previous
    assert result.startswith("[Security] Blocked:")
    assert "Docker is unavailable" in result
    return {"docker": "unavailable; fail-closed verified"}


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="corecoder-security-live-") as raw_tmp:
        tmp = Path(raw_tmp)
        report = {}
        report.update(await _exercise_network(tmp))
        report.update(await _exercise_shells(tmp))
        report.update(await _exercise_sandbox_failure())

    print("LIVE SECURITY SMOKE TEST: PASS")
    for name, outcome in report.items():
        print(f"- {name}: {outcome}")


if __name__ == "__main__":
    asyncio.run(main())
