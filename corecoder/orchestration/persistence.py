"""Secure local persistence helpers for LangGraph workflows."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

_ALLOWED_CHECKPOINT_TYPES = (
    ("corecoder.delegation", "WorkspaceMode"),
    ("corecoder.delegation", "TaskStatus"),
    ("corecoder.delegation", "TaskResult"),
    ("corecoder.orchestration.models", "FailureType"),
    ("corecoder.delegation", "TaskTestResult"),
    ("corecoder.orchestration.models", "ApprovalRequest"),
    ("corecoder.orchestration.models", "ReviewResult"),
    ("corecoder.orchestration.models", "ReviewVerdict"),
    ("corecoder.orchestration.models", "WorkflowStage"),
    ("corecoder.delegation", "TaskRole"),
    ("corecoder.orchestration.models", "WorkflowRequest"),
    ("corecoder.delegation", "TaskSpec"),
    ("corecoder.orchestration.models", "VerificationResult"),
    ("corecoder.delegation", "TaskUsage"),
    ("corecoder.delegation", "AcceptanceCheck"),
    # LangGraph stores this control-plane record when a workflow is paused.
    ("langgraph.types", "Interrupt"),
)


def _legacy_safe_ext_hook(code: int, data: bytes) -> Any:
    """Decode old checkpoint msgpack extensions without arbitrary imports."""
    import ormsgpack
    from langgraph.checkpoint.serde.jsonplus import (
        EXT_CONSTRUCTOR_KW_ARGS,
        EXT_CONSTRUCTOR_POS_ARGS,
        EXT_CONSTRUCTOR_SINGLE_ARG,
        EXT_PYDANTIC_V1,
        EXT_PYDANTIC_V2,
    )

    supported = {
        EXT_CONSTRUCTOR_SINGLE_ARG,
        EXT_CONSTRUCTOR_POS_ARGS,
        EXT_CONSTRUCTOR_KW_ARGS,
        EXT_PYDANTIC_V1,
        EXT_PYDANTIC_V2,
    }
    if code not in supported:
        raise ValueError(f"checkpoint extension code {code} is not allowlisted")
    values = ormsgpack.unpackb(
        data,
        ext_hook=_legacy_safe_ext_hook,
        option=ormsgpack.OPT_NON_STR_KEYS,
    )
    if not isinstance(values, (list, tuple)) or len(values) < 3:
        raise ValueError("invalid checkpoint extension payload")
    module_name, type_name, payload = values[:3]
    if (module_name, type_name) not in _ALLOWED_CHECKPOINT_TYPES:
        raise ValueError(
            f"checkpoint type {module_name}.{type_name} is not allowlisted"
        )
    cls = getattr(importlib.import_module(module_name), type_name)
    if code == EXT_CONSTRUCTOR_SINGLE_ARG:
        return cls(payload)
    if code == EXT_CONSTRUCTOR_POS_ARGS:
        return cls(*payload)
    if code == EXT_CONSTRUCTOR_KW_ARGS:
        return cls(**payload)
    if code == EXT_PYDANTIC_V1:
        return cls(**payload)
    return cls.model_validate(payload)


def safe_checkpoint_serializer() -> Any:
    """Return a non-pickle serializer restricted to CoreCoder state types."""
    try:
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    except ImportError as exc:
        raise RuntimeError(
            "CoreCoder installation is missing its LangGraph dependencies"
        ) from exc
    parameters = inspect.signature(JsonPlusSerializer).parameters
    if "allowed_msgpack_modules" in parameters:
        return JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_CHECKPOINT_TYPES)
    return JsonPlusSerializer(__unpack_ext_hook__=_legacy_safe_ext_hook)


@asynccontextmanager
async def encrypted_sqlite_checkpointer(
    path: str | Path,
    *,
    key: bytes,
) -> AsyncIterator[Any]:
    """Open an AES-encrypted SQLite checkpointer for local workflows.

    The encryption key is caller-owned and is never written next to the
    database. A 16, 24, or 32-byte AES key is required so a weak passphrase is
    not silently treated as encryption material.
    """
    if len(key) not in {16, 24, 32}:
        raise ValueError("LangGraph checkpoint AES key must be 16, 24, or 32 bytes")
    try:
        import aiosqlite
        from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    except ImportError as exc:
        raise RuntimeError(
            "CoreCoder installation is missing its LangGraph persistence dependencies"
        ) from exc

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    serializer = EncryptedSerializer.from_pycryptodome_aes(
        serde=safe_checkpoint_serializer(),
        key=key,
    )
    async with aiosqlite.connect(str(target)) as connection:
        saver = AsyncSqliteSaver(connection, serde=serializer)
        setup = getattr(saver, "setup", None)
        if setup is not None:
            await setup()
        yield saver
