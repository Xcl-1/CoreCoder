"""Append-only, corruption-tolerant persistence for delegated task state."""

from __future__ import annotations

import ctypes
import hashlib
import os
import socket
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .delegation import TaskEvent, TaskResult


class TaskLeaseRecord(BaseModel):
    """Small, non-sensitive identity record for one workspace scheduler."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    owner_id: str = Field(min_length=1, max_length=80)
    agent_id: str = Field(min_length=1, max_length=80)
    process_id: int = Field(gt=0)
    hostname: str = Field(min_length=1, max_length=255)
    acquired_at: float = Field(gt=0)


def _process_alive(process_id: int) -> bool:
    if process_id == os.getpid():
        return True
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel32.GetExitCodeProcess.restype = ctypes.c_int
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel32.OpenProcess(0x1000, 0, process_id)
        if not handle:
            # Access denied means the process exists but cannot be queried.
            return ctypes.get_last_error() != 87
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True
            return exit_code.value == 259
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except OSError as exc:
        return getattr(exc, "winerror", None) != 87
    return True


class TaskWorkspaceLease:
    """Atomic, heartbeat-backed single scheduler ownership for one workspace."""

    def __init__(
        self,
        root: str | Path,
        journal_id: str,
        agent_id: str,
        *,
        stale_after: float = 30.0,
        heartbeat_interval: float | None = None,
    ):
        if stale_after < 5:
            raise ValueError("task lease stale_after must be at least 5 seconds")
        interval = heartbeat_interval or min(5.0, stale_after / 3)
        if not 0 < interval < stale_after:
            raise ValueError("task lease heartbeat interval must be below stale_after")
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(journal_id.encode("utf-8", errors="replace")).hexdigest()
        self.path = self.root / f"tasks_{digest}.lease"
        self.record = TaskLeaseRecord(
            owner_id=f"owner_{uuid.uuid4().hex}",
            agent_id=agent_id,
            process_id=os.getpid(),
            hostname=socket.gethostname() or "unknown-host",
            acquired_at=time.time(),
        )
        self.stale_after = stale_after
        self.heartbeat_interval = interval
        self._owned = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def owns(self) -> bool:
        return self._owned and self._matches_owner()

    def owner(self) -> TaskLeaseRecord | None:
        return self._read_record()

    def acquire(self) -> bool:
        """Acquire an absent/stale lease atomically; never steal from a live PID."""
        with self._lock:
            if self.owns:
                return True
            for _ in range(3):
                try:
                    descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except FileExistsError:
                    if not self._reclaimable():
                        return False
                    stale_path = self.path.with_name(
                        f"{self.path.name}.stale-{self.record.owner_id}"
                    )
                    try:
                        self.path.replace(stale_path)
                    except FileNotFoundError:
                        continue
                    try:
                        stale_path.unlink()
                    except OSError:
                        pass
                    continue
                try:
                    os.write(descriptor, self.record.model_dump_json().encode("utf-8"))
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                self._owned = True
                self._stop.clear()
                self._thread = threading.Thread(
                    target=self._heartbeat,
                    name="corecoder-task-lease",
                    daemon=True,
                )
                self._thread.start()
                return True
            return False

    def close(self) -> None:
        """Release only this owner's lease; safe to call repeatedly."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self.heartbeat_interval + 0.5)
        with self._lock:
            if self._matches_owner():
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
            self._owned = False
            self._thread = None

    def _heartbeat(self) -> None:
        while not self._stop.wait(self.heartbeat_interval):
            with self._lock:
                if not self._matches_owner():
                    self._owned = False
                    return
                try:
                    os.utime(self.path, None)
                except OSError:
                    self._owned = False
                    return

    def _matches_owner(self) -> bool:
        current = self._read_record()
        return current is not None and current.owner_id == self.record.owner_id

    def _read_record(self) -> TaskLeaseRecord | None:
        try:
            return TaskLeaseRecord.model_validate_json(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None

    def _reclaimable(self) -> bool:
        current = self._read_record()
        try:
            stale = time.time() - self.path.stat().st_mtime > self.stale_after
        except FileNotFoundError:
            return True
        if current is None:
            return stale
        if current.hostname == self.record.hostname:
            return not _process_alive(current.process_id)
        return stale


class TaskJournalRecord(BaseModel):
    """One atomic lifecycle transition and its result, when available."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    event: TaskEvent
    result: TaskResult | None = None


@dataclass(frozen=True)
class TaskJournalLoad:
    records: tuple[TaskJournalRecord, ...] = ()
    invalid_lines: int = 0
    total_lines: int = 0


class TaskJournal:
    """Bounded JSONL journal keyed by a traversal-safe owner digest."""

    def __init__(
        self,
        root: str | Path,
        journal_id: str,
        *,
        max_records: int = 8_000,
    ):
        if not 8 <= max_records <= 80_000:
            raise ValueError("task journal max_records must be between 8 and 80000")
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(journal_id.encode("utf-8", errors="replace")).hexdigest()
        self.path = self.root / f"tasks_{digest}.jsonl"
        self.max_records = max_records
        self._recent: deque[str] = deque(maxlen=max_records)
        self._line_count = 0
        self._writes_since_compaction = 0
        self._lock = threading.Lock()

    def load(self) -> TaskJournalLoad:
        """Load recent valid records while tolerating a partial final write."""
        if not self.path.exists():
            return TaskJournalLoad()
        records: deque[TaskJournalRecord] = deque(maxlen=self.max_records)
        encoded: deque[str] = deque(maxlen=self.max_records)
        invalid = total = 0
        try:
            stream = self.path.open("r", encoding="utf-8", errors="replace")
        except OSError:
            return TaskJournalLoad(invalid_lines=1)
        with stream:
            for line in stream:
                if not line.strip():
                    continue
                total += 1
                try:
                    record = TaskJournalRecord.model_validate_json(line)
                except (ValueError, TypeError):
                    invalid += 1
                    continue
                records.append(record)
                encoded.append(record.model_dump_json())
        with self._lock:
            self._recent = deque(encoded, maxlen=self.max_records)
            self._line_count = total
            self._writes_since_compaction = 0
        return TaskJournalLoad(tuple(records), invalid, total)

    def record(self, event: TaskEvent, result: TaskResult | None = None) -> None:
        """Append one transition; objective and context are never accepted here."""
        record = TaskJournalRecord(event=event, result=result)
        encoded = record.model_dump_json()
        with self._lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(encoded + "\n")
            self._recent.append(encoded)
            self._line_count += 1
            self._writes_since_compaction += 1
            if (
                self._line_count > self.max_records
                and self._writes_since_compaction >= max(8, self.max_records // 10)
            ):
                self._compact_locked()

    def _compact_locked(self) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text("\n".join(self._recent) + "\n", encoding="utf-8")
        temporary.replace(self.path)
        self._line_count = len(self._recent)
        self._writes_since_compaction = 0
