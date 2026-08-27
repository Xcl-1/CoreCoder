"""Replay log — serialise every think→act→observe cycle as JSONL.

Each line is a complete `StepRecord` JSON object.  Append-only with an
explicit `flush()` after every write, so even if the agent crashes all
completed steps are on disk.

The log lives at ``~/.corecoder/replays/<session_id>.jsonl``, matching
the convention of ``~/.corecoder/sessions/`` for session persistence.
"""

import re
import time
from pathlib import Path

from .models import StepRecord

REPLAYS_DIR = Path.cwd() / "replays"
_SAFE_REPLAY_RE = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_REPLAY_ID_LEN = 100


def _normalize_replay_id(session_id: str | None) -> str:
    value = session_id or time.strftime("%Y%m%d_%H%M%S")
    name = value.strip().replace("\\", "/").split("/")[-1]
    name = _SAFE_REPLAY_RE.sub("-", name).strip(".-_")[:_MAX_REPLAY_ID_LEN]
    return name or time.strftime("%Y%m%d_%H%M%S")


class ReplayLogger:
    """Append-only JSONL logger.  One line per agent step."""

    def __init__(self, session_id: str | None = None):
        self.session_id = _normalize_replay_id(session_id)
        REPLAYS_DIR.mkdir(parents=True, exist_ok=True)
        self._path = (REPLAYS_DIR / f"{self.session_id}.jsonl").resolve()
        if self._path.parent != REPLAYS_DIR.resolve():
            raise ValueError("Invalid replay session id")
        self._file = None

    # -- context manager -------------------------------------------------

    def open(self):
        self._file = open(str(self._path), "a", encoding="utf-8")  # noqa: SIM115 — file stays open for appends

    def close(self):
        if self._file:
            self._file.close()
            self._file = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    # -- write -----------------------------------------------------------

    def log(self, record: StepRecord):
        """Append one step record as a JSON line, flushing immediately."""
        if self._file:
            self._file.write(record.model_dump_json() + "\n")
            self._file.flush()

    @property
    def path(self) -> Path:
        return self._path
