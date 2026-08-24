"""LLM-driven extraction of durable facts from a conversation."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from pydantic import TypeAdapter, ValidationError

from .models import ExtractedMemory, Memory

if TYPE_CHECKING:
    from ..llm import LLM

logger = logging.getLogger(__name__)
_RESULT_ADAPTER = TypeAdapter(list[ExtractedMemory])
_SECRET_RE = re.compile(
    r"(?:api[_ -]?key|password|access[_ -]?token|secret)\s*[:=]\s*\S+|\bsk-[A-Za-z0-9_-]{12,}",
    re.IGNORECASE,
)
_DURABLE_MARKERS = (
    "请记住",
    "记住",
    "以后",
    "今后",
    "始终",
    "总是",
    "一律",
    "默认",
    "长期",
    "我的偏好",
    "我偏好",
    "我喜欢",
    "不要再",
    "更新我的",
    "remember",
    "from now on",
    "always",
    "never",
    "by default",
    "my preference",
    "i prefer",
    "update my",
)
_LOCAL_MARKERS = (
    "这次",
    "本次",
    "当前任务",
    "这个任务",
    "只展示",
    "不要修改项目文件",
    "暂时",
    "for this task",
    "this time",
    "for now",
    "just show",
    "only show",
    "do not modify project files",
)


class MemoryExtractor:
    def __init__(self, llm: LLM, max_conversation_chars: int = 16_000):
        self.llm = llm
        self.max_conversation_chars = max_conversation_chars

    def extract(self, messages: list[dict], existing: list[Memory]) -> list[ExtractedMemory]:
        conversation = self._conversation(messages)
        if not conversation:
            return []
        known = [
            {
                "id": memory.id,
                "title": memory.title,
                "description": memory.description,
                "content": memory.content[:1_000],
                "type": memory.type,
                "scope": memory.scope,
                "keywords": memory.keywords,
            }
            for memory in existing[:30]
        ]
        prompt = f"""Analyze this coding-agent conversation and extract only durable information useful in future sessions.

Keep only:
- explicit user preferences;
- explicit corrections or feedback about how the agent should behave;
- stable project conventions or decisions;
- stable reference facts the user deliberately provided.

Do not keep temporary task instructions such as "for this task", "only show code", or "do not modify files" unless the user explicitly says they should apply in future sessions. Do not keep guesses made by the assistant, tool output, secrets, credentials, or information useful only in this session. Prefer no memory over a speculative memory.

Return ONLY a JSON array. Each item must contain:
{{"action":"create|merge|ignore","target_id":null,"title":"...","description":"...","content":"...","type":"user|feedback|project|reference","scope":"global|project","keywords":["..."],"confidence":0.0,"evidence":"exact quote from a user message"}}

Evidence must be copied exactly from a user message, never from an assistant message. Use action="merge" and target_id when the new information updates an existing memory. For a merge, content must be the complete replacement content. Use global scope only for explicit user-wide preferences; project facts must use project scope.

Existing memories:
{json.dumps(known, ensure_ascii=False)}

Conversation:
{conversation}
"""
        request = [
            {"role": "system", "content": "You extract conservative, structured long-term memory."},
            {"role": "user", "content": prompt},
        ]
        last_error: Exception | None = None
        for attempt in range(2):
            raw_output = ""
            try:
                response = self.llm.chat(messages=request, tools=None, on_token=None)
                raw_output = response.content
                extracted = self._parse(raw_output)
                return self._filter(extracted, messages)
            except (json.JSONDecodeError, ValidationError, AttributeError, TypeError, ValueError) as exc:
                last_error = exc
                if attempt == 0:
                    request.extend(
                        [
                            {"role": "assistant", "content": str(raw_output)[:5_000]},
                            {
                                "role": "user",
                                "content": (
                                    "Your output was invalid. Return only a JSON array using the exact schema above. "
                                    "Every non-ignore item must include an exact evidence quote from a user message."
                                ),
                            },
                        ]
                    )
            except Exception as exc:  # noqa: BLE001 - provider SDKs expose unrelated exception types
                logger.warning("Memory extraction request failed: %s", exc)
                return []
        logger.warning("Memory extraction failed after one repair attempt: %s", last_error)
        return []

    def _conversation(self, messages: list[dict]) -> str:
        parts: list[str] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
                continue
            parts.append(f"[{role}] {content.strip()}")
        flattened = "\n\n".join(parts)
        if not any(part.startswith("[user]") for part in parts):
            return ""
        return flattened[-self.max_conversation_chars :]

    @staticmethod
    def _parse(text: str) -> list[ExtractedMemory]:
        decoder = json.JSONDecoder()
        payload = None
        for index, character in enumerate(text):
            if character not in "[{":
                continue
            try:
                candidate, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, (list, dict)):
                payload = candidate
                break
        if payload is None:
            raise ValueError("No JSON value in memory extraction response")
        if isinstance(payload, dict):
            if isinstance(payload.get("memories"), list):
                payload = payload["memories"]
            elif isinstance(payload.get("items"), list):
                payload = payload["items"]
            elif "action" in payload:
                payload = [payload]
            else:
                raise ValueError("JSON object has no memory list")
        return _RESULT_ADAPTER.validate_python(payload)

    @staticmethod
    def _filter(extracted: list[ExtractedMemory], messages: list[dict]) -> list[ExtractedMemory]:
        user_text = "\n".join(
            str(message.get("content", ""))
            for message in messages
            if message.get("role") == "user" and isinstance(message.get("content"), str)
        )
        normalized_users = " ".join(user_text.casefold().split())
        safe: list[ExtractedMemory] = []
        for item in extracted[:10]:
            if item.action == "ignore":
                continue
            combined = " ".join([item.title, item.description, item.content, *item.keywords])
            if _SECRET_RE.search(combined):
                continue
            evidence = " ".join(item.evidence.strip(" \"'“”‘’").casefold().split())
            if not evidence or evidence not in normalized_users:
                continue
            has_durable_marker = any(marker in evidence for marker in _DURABLE_MARKERS)
            has_local_marker = any(marker in evidence for marker in _LOCAL_MARKERS)
            if has_local_marker and not has_durable_marker:
                continue
            if item.scope == "global" and item.type in ("user", "feedback") and not has_durable_marker:
                continue
            safe.append(item)
        return safe
