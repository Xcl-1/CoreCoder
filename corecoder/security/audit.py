"""Audit logger — JSONL record of every guarded tool call.

Each log entry captures *what* was attempted, *what was decided*,
and *why*.  Logs rotate daily; entries older than 30 days are
automatically pruned on each write.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

AUDIT_DIR = Path.home() / ".corecoder" / "audit"
_MAX_AGE_DAYS = 30


@dataclass
class AuditEntry:
    """One auditable tool-call event."""

    timestamp: str          # ISO-8601
    tool_name: str
    arguments_summary: str  # truncated to 200 chars
    decision: str           # "allow" | "deny" | "ask_denied"
    rule_source: str        # "user" | "project" | "builtin"
    reason: str
    user_confirmed: bool = False
    frequency_checked: bool = False
    frequency_passed: bool = True


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
        "decision": entry.decision,
        "rule_source": entry.rule_source,
        "reason": entry.reason,
        "user_confirmed": entry.user_confirmed,
        "frequency_checked": entry.frequency_checked,
        "frequency_passed": entry.frequency_passed,
    }
