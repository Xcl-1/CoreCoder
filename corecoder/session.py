"""Session persistence - save and resume conversations.

Claude Code maintains session state via QueryEngine (1295 lines).
CoreCoder distills this to: JSON dump of messages + model config.
"""

import json
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

SESSIONS_DIR = Path.home() / ".corecoder" / "sessions"
_SAFE_SESSION_RE = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_SESSION_ID_LEN = 100  # keep filenames comfortably under the OS limit
_SESSION_VERSION = 2


@dataclass(frozen=True)
class SessionRecord:
    """A saved session with model context and an immutable display transcript."""

    session_id: str
    model: str
    saved_at: str
    messages: list[dict]
    transcript: list[dict]
    version: int = _SESSION_VERSION


def _normalize_session_id(session_id: str | None) -> str:
    if not session_id:
        return _new_session_id()

    name = session_id.strip().replace("\\", "/").split("/")[-1]
    name = _SAFE_SESSION_RE.sub("-", name).strip(".-_")
    if len(name) > _MAX_SESSION_ID_LEN:
        name = name[:_MAX_SESSION_ID_LEN].strip(".-_")
    return name or _new_session_id()


def _new_session_id() -> str:
    return f"session_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _session_path(session_id: str) -> Path:
    path = (SESSIONS_DIR / f"{_normalize_session_id(session_id)}.json").resolve()
    root = SESSIONS_DIR.resolve()
    if root != path.parent:
        raise ValueError("Invalid session id")
    return path


def derive_transcript(messages: list[dict]) -> list[dict]:
    """Derive a clean user-facing transcript from a legacy message list."""
    transcript: list[dict] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str) or not content:
            continue
        transcript.append({"role": role, "content": content})
    return transcript


def save_session(
    messages: list[dict],
    model: str,
    session_id: str | None = None,
    transcript: list[dict] | None = None,
) -> str:
    """Save conversation to disk. Returns the session ID."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    session_id = _normalize_session_id(session_id)

    data = {
        "version": _SESSION_VERSION,
        "id": session_id,
        "model": model,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "messages": messages,
        "transcript": derive_transcript(messages if transcript is None else transcript),
    }

    path = _session_path(session_id)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return session_id


def load_session(session_id: str) -> tuple[list[dict], str] | None:
    """Load a saved session. Returns (messages, model) or None."""
    record = load_session_record(session_id)
    if record is None:
        return None
    return record.messages, record.model


def load_session_record(session_id: str) -> SessionRecord | None:
    """Load the complete record, deriving a transcript for version-1 sessions."""
    path = _session_path(session_id)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        messages = data["messages"]
        model = data["model"]
        if not isinstance(messages, list) or not isinstance(model, str):
            return None
        transcript = data.get("transcript")
        if not isinstance(transcript, list):
            transcript = derive_transcript(messages)
        return SessionRecord(
            session_id=str(data.get("id") or _normalize_session_id(session_id)),
            model=model,
            saved_at=str(data.get("saved_at") or "?"),
            messages=messages,
            transcript=derive_transcript(transcript),
            version=int(data.get("version") or 1),
        )
    except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError):
        # a corrupt or truncated session file shouldn't crash resume
        return None


def list_sessions() -> list[dict]:
    """List available sessions, newest first."""
    if not SESSIONS_DIR.exists():
        return []

    sessions = []
    for f in SESSIONS_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                continue
            # Prefer the immutable transcript; legacy sessions fall back to messages.
            preview = ""
            history = data.get("transcript")
            if not isinstance(history, list):
                history = data.get("messages", [])
            if not isinstance(history, list):
                history = []
            for m in history:
                if not isinstance(m, dict):
                    continue
                if m.get("role") == "user" and m.get("content"):
                    preview = str(m["content"])[:80]
                    break
            sessions.append({
                "id": str(data.get("id") or f.stem),
                "model": str(data.get("model") or "?"),
                "saved_at": str(data.get("saved_at") or "?"),
                "preview": preview,
            })
        except (json.JSONDecodeError, KeyError, OSError, TypeError):
            continue

    sessions.sort(key=lambda item: item["saved_at"], reverse=True)
    return sessions
