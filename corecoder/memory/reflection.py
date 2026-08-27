"""Execution reflection built from conversation messages and replay records."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from .models import SessionReflection

if TYPE_CHECKING:
    from ..llm import LLM

logger = logging.getLogger(__name__)
_SECRET_RE = re.compile(
    r"(?:api[_ -]?key|password|access[_ -]?token|secret)[\"']?\s*[:=]\s*[\"']?\S+|\bsk-[A-Za-z0-9_-]{12,}",
    re.IGNORECASE,
)


def redact_secrets(text: str) -> str:
    return _SECRET_RE.sub("[REDACTED]", text)


class MemoryReflector:
    """Ask the LLM for a conservative, evidence-backed execution review."""

    def __init__(self, llm: LLM, max_source_chars: int = 24_000):
        self.llm = llm
        self.max_source_chars = max_source_chars

    def reflect(
        self,
        messages: list[dict],
        replay_path: Path | str | None = None,
    ) -> SessionReflection | None:
        source = self.source_text(messages, replay_path)
        if not source:
            return None
        execution_stats = self._execution_stats(replay_path)

        prompt = f"""Review this coding-agent execution and produce a concise, evidence-backed reflection.

Determine the actual outcome from verification evidence, not from confident language. Capture failed attempts, root causes, effective actions, verification, and reusable lessons. Do not include credentials or secrets. Evidence and verification entries must be short exact quotes copied from the source. If success was not verified, use partial or unknown rather than success.

Treat repeated shell syntax mistakes, platform mismatches, blocked probes, and abandoned commands as execution noise unless the source proves a reusable root cause and a verified remedy. Do not turn a simple acknowledgement or a request to remember a preference into an execution lesson. Never claim a root cause from one failed command alone.

Return ONLY one JSON object with this schema:
{{"task_summary":"...","outcome":"success|partial|failure|unknown","summary":"...","failures":["..."],"root_causes":["..."],"effective_actions":["..."],"verification":["..."],"reusable_lessons":["..."],"evidence":["exact source quote"]}}

Deterministic execution facts:
{json.dumps(execution_stats)}

Execution source:
{source}
"""
        request = [
            {"role": "system", "content": "You are a conservative execution reviewer."},
            {"role": "user", "content": prompt},
        ]
        raw_output = ""
        for attempt in range(2):
            try:
                response = self.llm.chat(messages=request, tools=None, on_token=None)
                raw_output = response.content
                reflection = self._parse(raw_output)
                validated = self._validate_evidence(reflection, source)
                return validated.model_copy(update=execution_stats)
            except (json.JSONDecodeError, ValidationError, AttributeError, TypeError, ValueError) as exc:
                if attempt == 0:
                    request.extend([
                        {"role": "assistant", "content": str(raw_output)[:5_000]},
                        {
                            "role": "user",
                            "content": "Return only one valid JSON object using the exact schema.",
                        },
                    ])
                    continue
                logger.warning("Execution reflection failed after one repair attempt: %s", exc)
            except Exception as exc:  # noqa: BLE001 - provider exceptions vary
                logger.warning("Execution reflection request failed: %s", exc)
                return None
        return None

    def source_text(self, messages: list[dict], replay_path: Path | str | None = None) -> str:
        parts: list[str] = []
        for message in messages:
            role = message.get("role", "?")
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            limit = 1_500 if role == "tool" else 3_000
            parts.append(f"[{role}] {redact_secrets(content.strip()[:limit])}")

        replay = self._read_replay(replay_path)
        if replay:
            parts.append(f"[replay]\n{replay}")
        return "\n\n".join(parts)[-self.max_source_chars :]

    @staticmethod
    def _read_replay(replay_path: Path | str | None) -> str:
        if not replay_path:
            return ""
        path = Path(replay_path)
        if not path.exists():
            return ""
        rows: list[str] = []
        seen: set[str] = set()
        omitted_duplicates = 0
        try:
            for line in path.read_text(encoding="utf-8").splitlines()[-50:]:
                if not line.strip():
                    continue
                record = json.loads(line)
                step = record.get("step", "?")
                response = record.get("llm_response", {}).get("content", "")
                if response:
                    rows.append(f"step {step} response: {redact_secrets(str(response)[:1_000])}")
                for execution in record.get("tool_executions", []):
                    name = execution.get("name", "unknown")
                    success = execution.get("success", False)
                    arguments = redact_secrets(
                        json.dumps(execution.get("arguments", {}), ensure_ascii=False)[:1_000]
                    )
                    result = redact_secrets(str(execution.get("result", ""))[:1_500])
                    error = redact_secrets(str(execution.get("error", ""))[:500])
                    row = (
                        f"step {step} tool {name} arguments={arguments} success={success} "
                        f"error={error!r} result={result!r}"
                    )
                    signature = json.dumps(
                        [name, arguments, success, error, result],
                        ensure_ascii=False,
                        sort_keys=True,
                    ).casefold()
                    if signature in seen:
                        omitted_duplicates += 1
                        continue
                    seen.add(signature)
                    rows.append(row)
        except (OSError, json.JSONDecodeError, TypeError, AttributeError):
            logger.debug("Could not read replay for memory reflection", exc_info=True)
            return ""
        if omitted_duplicates:
            rows.append(f"[replay summary] omitted {omitted_duplicates} duplicate tool execution(s)")
        return "\n".join(rows[-25:])[-12_000:]

    @staticmethod
    def _execution_stats(replay_path: Path | str | None) -> dict[str, int]:
        stats = {"tool_executions": 0, "successful_tools": 0, "failed_tools": 0}
        if not replay_path:
            return stats
        path = Path(replay_path)
        if not path.exists():
            return stats
        try:
            for line in path.read_text(encoding="utf-8").splitlines()[-50:]:
                if not line.strip():
                    continue
                record = json.loads(line)
                for execution in record.get("tool_executions", []):
                    stats["tool_executions"] += 1
                    key = "successful_tools" if execution.get("success", False) else "failed_tools"
                    stats[key] += 1
        except (OSError, json.JSONDecodeError, TypeError, AttributeError):
            logger.debug("Could not calculate replay execution statistics", exc_info=True)
            return {"tool_executions": 0, "successful_tools": 0, "failed_tools": 0}
        return stats

    @staticmethod
    def _parse(text: str) -> SessionReflection:
        decoder = json.JSONDecoder()
        for index, character in enumerate(text):
            if character != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return SessionReflection.model_validate(payload)
        raise ValueError("No JSON reflection object found")

    @staticmethod
    def _validate_evidence(reflection: SessionReflection, source: str) -> SessionReflection:
        normalized_source = " ".join(source.casefold().split())
        def valid_quotes(quotes: list[str], limit: int) -> list[str]:
            valid = []
            for quote in quotes[:limit]:
                cleaned = quote.strip()[:500]
                normalized = " ".join(cleaned.casefold().split())
                if normalized and normalized in normalized_source and not _SECRET_RE.search(cleaned):
                    valid.append(cleaned)
            return valid

        evidence = valid_quotes(reflection.evidence, 20)
        verification = valid_quotes(reflection.verification, 10)
        return reflection.model_copy(update={"evidence": evidence, "verification": verification})
