"""Audit logger — JSONL record of every guarded tool call.

Each log entry captures *what* was attempted, *what was decided*,
and *why*.  Logs rotate daily; entries older than 30 days are
automatically pruned on each write.
"""

from __future__ import annotations

import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

AUDIT_DIR = Path.home() / ".corecoder" / "audit"
_MAX_AGE_DAYS = 30


@dataclass
class AuditEntry:
    """One auditable tool-call or control-plane event."""

    timestamp: str          # ISO-8601
    tool_name: str
    arguments_summary: str  # truncated to 200 chars
    decision: str           # "allow" | "deny" | "flag" | "policy"
    rule_source: str        # "user" | "project" | "builtin"
    reason: str
    arguments_digest: str = ""
    user_confirmed: bool = False
    frequency_checked: bool = False
    frequency_passed: bool = True
    risk_level: str = ""
    risk_reasons: list[str] = field(default_factory=list)
    capability_scope: str = ""
    network_access: str = ""
    declared_risk: str = ""
    network_policy_action: str = ""
    network_destinations: list[str] = field(default_factory=list)
    agent_id: str = ""
    parent_id: str = ""
    task_id: str = ""
    permission_scope: str = ""
    workspace_mode: str = ""
    event_type: str = "tool_call"
    event_sequence: int = 0


@dataclass(frozen=True)
class AuditQueryResult:
    """Bounded, corruption-tolerant view of one day's audit records."""

    entries: tuple[dict, ...] = ()
    total_entries: int = 0
    total_matches: int = 0
    invalid_lines: int = 0
    decision_counts: dict[str, int] = field(default_factory=dict)
    risk_counts: dict[str, int] = field(default_factory=dict)
    confirmed_count: int = 0


class AuditLogger:
    """Append-only JSONL audit log, one file per day."""

    def __init__(self, log_dir: Path | None = None):
        self._dir = (log_dir or AUDIT_DIR).expanduser()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._last_date: str = ""

    # ---- public -----------------------------------------------------------

    def log(self, entry: AuditEntry) -> None:
        """Append one entry to today's log file.  Prunes old files."""
        today = time.strftime("%Y-%m-%d")
        if today != self._last_date:
            self._last_date = today
            self._prune()

        path = self._dir / f"audit_{today}.jsonl"
        with open(str(path), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(_entry_to_dict(entry), ensure_ascii=False) + "\n")

    def query(
        self,
        *,
        date: str | None = None,
        decisions: set[str] | None = None,
        tool_name: str | None = None,
        event_type: str | None = None,
        task_id: str | None = None,
        agent_id: str | None = None,
        confirmed_only: bool = False,
        limit: int = 10,
    ) -> AuditQueryResult:
        """Read recent matching records without exposing log-file internals.

        Invalid JSON lines are counted and skipped so one partial/corrupt write
        does not make the rest of the day's security history unavailable.
        """
        day = date or time.strftime("%Y-%m-%d")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
            raise ValueError("audit date must use YYYY-MM-DD")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 1000:
            raise ValueError("audit query limit must be between 1 and 1000")

        path = self._dir / f"audit_{day}.jsonl"
        if not path.exists():
            return AuditQueryResult()

        recent: deque[dict] = deque(maxlen=limit)
        total_entries = total_matches = invalid_lines = confirmed_count = 0
        decision_counts: dict[str, int] = {}
        risk_counts: dict[str, int] = {}
        try:
            stream = path.open("r", encoding="utf-8", errors="replace")
        except OSError:
            return AuditQueryResult(invalid_lines=1)

        with stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    invalid_lines += 1
                    continue
                if not isinstance(entry, dict):
                    invalid_lines += 1
                    continue

                total_entries += 1
                decision = str(entry.get("decision", "unknown"))
                risk = str(entry.get("risk_level", "") or "none")
                decision_counts[decision] = decision_counts.get(decision, 0) + 1
                risk_counts[risk] = risk_counts.get(risk, 0) + 1
                if entry.get("user_confirmed") is True:
                    confirmed_count += 1

                matches = (
                    (decisions is None or decision in decisions)
                    and (tool_name is None or entry.get("tool_name") == tool_name)
                    and (event_type is None or entry.get("event_type", "tool_call") == event_type)
                    and (task_id is None or entry.get("task_id") == task_id)
                    and (agent_id is None or entry.get("agent_id") == agent_id)
                    and (not confirmed_only or entry.get("user_confirmed") is True)
                )
                if matches:
                    total_matches += 1
                    recent.append(entry)

        return AuditQueryResult(
            entries=tuple(recent),
            total_entries=total_entries,
            total_matches=total_matches,
            invalid_lines=invalid_lines,
            decision_counts=decision_counts,
            risk_counts=risk_counts,
            confirmed_count=confirmed_count,
        )

    # ---- internal ---------------------------------------------------------

    def _prune(self) -> None:
        """Remove log files older than ``_MAX_AGE_DAYS`` days."""
        cutoff = time.time() - _MAX_AGE_DAYS * 86400
        for f in self._dir.glob("audit_*.jsonl"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
            except OSError:
                pass


def _entry_to_dict(entry: AuditEntry) -> dict:
    return {
        "timestamp": entry.timestamp,
        "tool_name": entry.tool_name,
        "arguments_summary": entry.arguments_summary,
        "arguments_digest": entry.arguments_digest,
        "decision": entry.decision,
        "rule_source": entry.rule_source,
        "reason": entry.reason,
        "user_confirmed": entry.user_confirmed,
        "frequency_checked": entry.frequency_checked,
        "frequency_passed": entry.frequency_passed,
        "risk_level": entry.risk_level,
        "risk_reasons": entry.risk_reasons,
        "capability_scope": entry.capability_scope,
        "network_access": entry.network_access,
        "declared_risk": entry.declared_risk,
        "network_policy_action": entry.network_policy_action,
        "network_destinations": entry.network_destinations,
        "agent_id": entry.agent_id,
        "parent_id": entry.parent_id,
        "task_id": entry.task_id,
        "permission_scope": entry.permission_scope,
        "workspace_mode": entry.workspace_mode,
        "event_type": entry.event_type,
        "event_sequence": entry.event_sequence,
    }
