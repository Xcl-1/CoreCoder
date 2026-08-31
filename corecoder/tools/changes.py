"""Session-scoped file snapshots for safe, conflict-aware undo."""

from __future__ import annotations

import contextvars
import os
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FileSnapshot:
    path: Path
    original: bytes | None
    expected_current: bytes
    original_mode: int | None = None
    created_dirs: set[Path] = field(default_factory=set)


@dataclass
class UndoResult:
    restored: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def changed(self) -> int:
        return len(self.restored) + len(self.deleted)


class ChangeTracker:
    """Remember each file's pre-session bytes and latest tool-written bytes."""

    def __init__(self):
        self._snapshots: dict[str, FileSnapshot] = {}
        self._lock = threading.RLock()
        self.changed_files: set[str] = set()

    def record(
        self,
        path: Path | str,
        *,
        before: bytes | None,
        after: bytes,
        original_mode: int | None = None,
        created_dirs: set[Path] | None = None,
    ) -> None:
        resolved = Path(path).expanduser().resolve()
        key = str(resolved)
        with self._lock:
            existing = self._snapshots.get(key)
            if existing is None:
                self._snapshots[key] = FileSnapshot(
                    path=resolved,
                    original=before,
                    expected_current=after,
                    original_mode=original_mode,
                    created_dirs=set(created_dirs or ()),
                )
            else:
                existing.expected_current = after
                existing.created_dirs.update(created_dirs or ())
            self.changed_files.add(key)

    def undo_all(self, *, force: bool = False) -> UndoResult:
        """Restore tracked files, refusing to overwrite unexpected external edits."""
        result = UndoResult()
        removable_dirs: set[Path] = set()
        with self._lock:
            for key, snapshot in list(reversed(self._snapshots.items())):
                try:
                    current = snapshot.path.read_bytes() if snapshot.path.is_file() else None
                    if current != snapshot.expected_current and not force:
                        result.conflicts.append(key)
                        continue

                    if snapshot.original is None:
                        if snapshot.path.exists():
                            if not snapshot.path.is_file():
                                result.conflicts.append(key)
                                continue
                            snapshot.path.unlink()
                        result.deleted.append(key)
                    else:
                        snapshot.path.parent.mkdir(parents=True, exist_ok=True)
                        self._atomic_restore(snapshot)
                        result.restored.append(key)

                    removable_dirs.update(snapshot.created_dirs)
                    self._snapshots.pop(key, None)
                    self.changed_files.discard(key)
                except OSError as exc:
                    result.errors.append(f"{key}: {exc}")

            for directory in sorted(removable_dirs, key=lambda item: len(item.parts), reverse=True):
                try:
                    directory.rmdir()
                except (FileNotFoundError, OSError):
                    pass
        return result

    def clear(self) -> None:
        """Forget undo history without touching files."""
        with self._lock:
            self._snapshots.clear()
            self.changed_files.clear()

    def __len__(self) -> int:
        return len(self._snapshots)

    @staticmethod
    def _atomic_restore(snapshot: FileSnapshot) -> None:
        temporary = snapshot.path.with_name(
            f".{snapshot.path.name}.corecoder-undo-{uuid.uuid4().hex}.tmp"
        )
        try:
            temporary.write_bytes(snapshot.original or b"")
            if snapshot.original_mode is not None:
                os.chmod(temporary, snapshot.original_mode)
            os.replace(temporary, snapshot.path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


_default_tracker = ChangeTracker()
_active_tracker: contextvars.ContextVar[ChangeTracker | None] = contextvars.ContextVar(
    "corecoder_change_tracker", default=None
)


def current_change_tracker() -> ChangeTracker:
    tracker = _active_tracker.get()
    return tracker if tracker is not None else _default_tracker


def bind_change_tracker(tracker: ChangeTracker):
    return _active_tracker.set(tracker)


def reset_change_tracker(token) -> None:
    _active_tracker.reset(token)


def default_change_tracker() -> ChangeTracker:
    return _default_tracker


def missing_parent_dirs(path: Path) -> set[Path]:
    """Return parent directories that do not exist before a write."""
    missing: set[Path] = set()
    current = path.parent
    while not current.exists() and current != current.parent:
        missing.add(current)
        current = current.parent
    return missing
