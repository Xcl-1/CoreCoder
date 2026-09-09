"""Git worktree execution backend for centrally controlled child agents."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .tools.changes import ChangeTracker


class WorktreeError(RuntimeError):
    """Raised when an isolated workspace cannot be created or merged safely."""


def _git(cwd: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(cwd), *args],
            input=input_bytes,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise WorktreeError(f"git is unavailable: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise WorktreeError(f"git {' '.join(args)} failed: {detail or completed.returncode}")
    return completed.stdout


@dataclass
class WorktreeSession:
    """One detached worktree whose delta can be checked and merged centrally."""

    repo_root: Path
    path: Path
    task_id: str
    source_relative: Path = Path(".")
    retained: bool = False

    @property
    def working_root(self) -> Path:
        """Directory corresponding to the parent's original working root."""
        return (self.path / self.source_relative).resolve()

    @classmethod
    def create(cls, task_id: str, *, cwd: Path | None = None) -> WorktreeSession:
        source = (cwd or Path.cwd()).resolve()
        root_text = _git(source, "rev-parse", "--show-toplevel").decode(
            "utf-8", errors="replace"
        ).strip()
        root = Path(root_text).resolve()
        try:
            source_relative = source.relative_to(root)
        except ValueError as exc:
            raise WorktreeError("workspace root is outside the discovered repository") from exc
        if _git(root, "status", "--porcelain", "--untracked-files=all").strip():
            raise WorktreeError(
                "worktree mode requires a clean parent worktree so the child baseline is unambiguous"
            )

        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", task_id)[:80]
        if not safe_id or safe_id in {".", ".."}:
            raise WorktreeError("invalid task id for worktree path")
        base = (root / ".corecoder" / "worktrees").resolve()
        path = (base / safe_id).resolve()
        if path.parent != base:
            raise WorktreeError("worktree path escaped its managed root")
        if path.exists():
            raise WorktreeError(f"managed worktree path already exists: {path}")
        base.mkdir(parents=True, exist_ok=True)
        _git(root, "worktree", "add", "--detach", str(path), "HEAD")
        return cls(
            repo_root=root,
            path=path,
            task_id=task_id,
            source_relative=source_relative,
        )

    def collect_patch(self) -> tuple[bytes, tuple[str, ...]]:
        """Return a binary patch and validated repo-relative changed paths."""
        # Intent-to-add makes untracked files appear in git diff without
        # committing or sharing them with the parent index.
        _git(self.path, "add", "-N", "--", ".")
        patch = _git(
            self.path,
            "diff",
            "HEAD",
            "--binary",
            "--full-index",
            "--no-ext-diff",
            "--",
        )
        names_raw = _git(self.path, "diff", "HEAD", "--name-only", "-z", "--")
        names = []
        for raw in names_raw.split(b"\0"):
            if not raw:
                continue
            name = raw.decode("utf-8", errors="strict")
            candidate = Path(name)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise WorktreeError(f"unsafe path in worktree diff: {name}")
            target = (self.repo_root / candidate).resolve()
            if not (target == self.repo_root or target.is_relative_to(self.repo_root)):
                raise WorktreeError(f"worktree diff escaped repository: {name}")
            names.append(candidate.as_posix())
        return patch, tuple(names)

    def merge(self, tracker: ChangeTracker) -> tuple[str, ...]:
        """Check then apply the child delta to the parent and record undo data."""
        patch, names = self.collect_patch()
        if not patch:
            return ()
        before: dict[str, tuple[bytes | None, int | None]] = {}
        for name in names:
            target = self.repo_root / name
            before[name] = (
                target.read_bytes() if target.is_file() else None,
                target.stat().st_mode if target.is_file() else None,
            )

        _git(self.repo_root, "apply", "--check", "--whitespace=nowarn", "-", input_bytes=patch)
        _git(self.repo_root, "apply", "--whitespace=nowarn", "-", input_bytes=patch)

        for name in names:
            target = self.repo_root / name
            original, original_mode = before[name]
            current = target.read_bytes() if target.is_file() else None
            tracker.record(
                target,
                before=original,
                after=current,
                original_mode=original_mode,
            )
        return names

    def cleanup(self) -> None:
        """Remove only this exact, controller-created worktree."""
        if self.retained:
            return
        base = (self.repo_root / ".corecoder" / "worktrees").resolve()
        resolved = self.path.resolve()
        if resolved.parent != base:
            raise WorktreeError("refusing to remove a worktree outside the managed root")
        if resolved.exists():
            _git(self.repo_root, "worktree", "remove", "--force", str(resolved))
