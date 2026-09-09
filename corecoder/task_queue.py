"""Authenticated encrypted storage for explicitly durable delegated tasks."""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Literal

from cryptography.fernet import Fernet, InvalidToken
from pydantic import BaseModel, ConfigDict

from .delegation import TaskSpec


class DurableTaskEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    queued_at: float
    spec: TaskSpec


@dataclass(frozen=True)
class DurableQueueLoad:
    specs: tuple[TaskSpec, ...] = ()
    invalid_items: int = 0
    truncated: bool = False


class DurableTaskQueue:
    """One encrypted file per task, scoped to a workspace digest."""

    def __init__(
        self,
        root: str | Path,
        queue_id: str,
        *,
        key: str | bytes | None = None,
        max_items: int = 1_000,
    ):
        if not 1 <= max_items <= 10_000:
            raise ValueError("durable queue max_items must be between 1 and 10000")
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(queue_id.encode("utf-8", errors="replace")).hexdigest()
        self.path = self.root / f"queue_{digest}"
        self.path.mkdir(parents=True, exist_ok=True)
        self.key_path = self.root / ".task-queue.key"
        self.max_items = max_items
        material = key.encode("ascii") if isinstance(key, str) else key
        self._fernet = Fernet(material or self._load_or_create_key())

    def enqueue(self, spec: TaskSpec) -> Path:
        if not spec.durable:
            raise ValueError("only durable TaskSpec values may enter the durable queue")
        target = self._path(spec.task_id)
        if not target.exists() and len(list(islice(self.path.glob("*.task"), self.max_items))) >= self.max_items:
            raise RuntimeError("durable task queue is full")
        envelope = DurableTaskEnvelope(queued_at=time.time(), spec=spec)
        token = self._fernet.encrypt(envelope.model_dump_json().encode("utf-8"))
        temporary = target.with_suffix(f".tmp-{os.getpid()}")
        temporary.write_bytes(token)
        temporary.replace(target)
        return target

    def remove(self, task_id: str) -> bool:
        target = self._path(task_id)
        try:
            target.unlink()
        except FileNotFoundError:
            return False
        return True

    def load(self) -> DurableQueueLoad:
        specs: list[TaskSpec] = []
        invalid = 0
        items = list(islice(self.path.glob("*.task"), self.max_items + 1))
        truncated = len(items) > self.max_items
        for item in sorted(items[:self.max_items]):
            try:
                payload = self._fernet.decrypt(item.read_bytes())
                envelope = DurableTaskEnvelope.model_validate_json(payload)
                if not envelope.spec.durable or item != self._path(envelope.spec.task_id):
                    raise ValueError("durable queue identity mismatch")
            except (OSError, InvalidToken, ValueError, TypeError):
                invalid += 1
                continue
            specs.append(envelope.spec)
        return DurableQueueLoad(tuple(specs), invalid, truncated)

    def contains(self, task_id: str) -> bool:
        return self._path(task_id).exists()

    def _path(self, task_id: str) -> Path:
        digest = hashlib.sha256(task_id.encode("utf-8", errors="replace")).hexdigest()
        return self.path / f"{digest}.task"

    def _load_or_create_key(self) -> bytes:
        try:
            return self.key_path.read_bytes().strip()
        except FileNotFoundError:
            generated = Fernet.generate_key()
            try:
                descriptor = os.open(self.key_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                return self.key_path.read_bytes().strip()
            try:
                os.write(descriptor, generated + b"\n")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            try:
                self.key_path.chmod(0o600)
            except OSError:
                pass
            return generated
