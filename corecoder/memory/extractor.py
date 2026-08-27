"""LLM-driven extraction of durable facts from a conversation."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from pydantic import TypeAdapter, ValidationError

from .models import ExtractedMemory, Memory, SessionReflection

if TYPE_CHECKING:
    from ..llm import LLM

logger = logging.getLogger(__name__)
_RESULT_ADAPTER = TypeAdapter(list[ExtractedMemory])
_SECRET_RE = re.compile(
    r"(?:api[_ -]?key|password|access[_ -]?token|secret)[\"']?\s*[:=]\s*[\"']?\S+|\bsk-[A-Za-z0-9_-]{12,}",
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
_EXPLICIT_DURABLE_MARKERS = tuple(
    marker for marker in _DURABLE_MARKERS if marker not in ("以后", "今后")
)
_USER_PREFERENCE_MARKERS = (
    "请记住",
    "记住",
    "我的偏好",
    "我偏好",
    "我喜欢",
    "不要再",
    "更新我的",
    "始终",
    "总是",
    "一律",
    "默认",
    "remember",
    "from now on",
    "always",
    "never",
    "by default",
    "my preference",
    "i prefer",
    "update my",
)
_FUTURE_PREFERENCE_RE = re.compile(
    r"(?:以后|今后).{0,40}(?:回答|回复|解释|沟通|保持|优先|不要|称呼|语言)",
    re.IGNORECASE,
)
_PROFILE_RE = re.compile(
    r"(?:^|[，。！？,;]\s*)(?:我是|我在|我的职业|我的角色|我主要|i am|i'm|i work as|my role)",
    re.IGNORECASE,
)
_FEEDBACK_RE = re.compile(
    r"(?:你|你的|回答|回复).{0,30}(?:应该|需要|太|不够|错误|改成|不要)|"
    r"(?:you|your|answer|response).{0,40}(?:should|too|not enough|wrong|stop|change)",
    re.IGNORECASE,
)
_TASK_REQUEST_RE = re.compile(
    r"^\s*(?:请|请帮我|帮我|麻烦)?\s*(?:运行|执行|分析|总结|生成|创建|修复|实现|修改|查看|检查|测试|规划|解释|列出|打开|读取)|"
    r"^\s*(?:please\s+)?(?:run|execute|analy[sz]e|summarize|generate|create|fix|implement|modify|"
    r"check|test|plan|explain|list|open|read)\b",
    re.IGNORECASE,
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
        self.last_succeeded = False
        self.last_fallback_succeeded = False
        self.last_error: str | None = None

    def extract(
        self,
        messages: list[dict],
        existing: list[Memory],
        reflection: SessionReflection | None = None,
        evidence_source: str = "",
    ) -> list[ExtractedMemory]:
        self.last_succeeded = False
        self.last_error = None
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
                "status": memory.status,
                "version": memory.version,
            }
            for memory in existing[:30]
        ]
        reflection_payload = reflection.model_dump() if reflection else None
        prompt = f"""Analyze this coding-agent conversation and extract only durable information useful in future sessions.

Keep only:
- explicit user preferences;
- explicit corrections or feedback about how the agent should behave;
- stable project conventions or decisions;
- stable reference facts the user deliberately provided.
- verified reusable procedures derived from successful execution reflection;
- useful episodic records of a concrete task, outcome, failure, and fix;
- explicit user profile facts that should apply beyond one task.

Do not keep temporary task instructions such as "run this command", "analyze the result", "summarize reusable steps", "for this task", "only show code", or "do not modify files" unless the user separately states a durable preference, profile fact, or project convention. A request to produce a reusable procedure is not itself a user preference; preserve the verified procedure from execution reflection instead. Do not keep guesses made by the assistant, raw verbose tool output, secrets, credentials, or information useful only in this session. Tool results may support a concise procedure or episode but must not be copied wholesale. Prefer no memory over a speculative memory.

Only create a project-scoped procedure when the reflection outcome is success, at least one tool succeeded, and exact verification evidence is present. Only create a project-scoped episode when a real tool execution failed and the reflection identifies a useful failure or root-cause lesson; do not create episodes for routine successful work or preference acknowledgements. Profile memories require explicit user evidence. Use action="archive" with target_id when a previous memory is explicitly obsolete. Set supersedes when a new memory replaces an older one.

Return ONLY a JSON array. Each item must contain:
{{"action":"create|merge|archive|ignore","target_id":null,"title":"...","description":"...","content":"...","type":"user|feedback|project|reference|procedure|episode|profile","scope":"global|project","keywords":["..."],"confidence":0.0,"evidence":"exact source quote","supersedes":null}}

For user, feedback, profile, project, and reference memories, evidence must be copied exactly from a user message. Procedure and episode evidence may be copied from execution evidence. Use action="merge" and target_id when new information updates an existing memory. For a merge, content must be the complete replacement content. Use global scope only for explicit user-wide preferences; project execution knowledge must use project scope.

Existing memories:
{json.dumps(known, ensure_ascii=False)}

Execution reflection:
{json.dumps(reflection_payload, ensure_ascii=False)}

Conversation:
{conversation}
"""
        request = [
            {"role": "system", "content": "You extract conservative, structured long-term memory."},
            {"role": "user", "content": prompt},
        ]
        result = self._request(request, messages, reflection, evidence_source)
        self.last_succeeded = result is not None
        return result or []

    def extract_procedure_fallback(
        self,
        messages: list[dict],
        existing: list[Memory],
        reflection: SessionReflection,
        evidence_source: str,
    ) -> list[ExtractedMemory]:
        """Run a constrained second pass when a verified success produced no procedure."""
        return self.extract_execution_fallback(
            messages,
            existing,
            reflection,
            evidence_source,
            include_procedure=True,
            include_episode=False,
        )

    def extract_execution_fallback(
        self,
        messages: list[dict],
        existing: list[Memory],
        reflection: SessionReflection,
        evidence_source: str,
        *,
        include_procedure: bool = True,
        include_episode: bool = True,
    ) -> list[ExtractedMemory]:
        """Extract only evidence-backed execution assets after a general-pass miss or failure."""
        self.last_fallback_succeeded = False
        allowed_types: list[str] = []
        if include_procedure and self.supports_procedure(reflection):
            allowed_types.append("procedure")
        if include_episode and self.supports_episode(reflection):
            allowed_types.append("episode")
        if not allowed_types:
            return []
        conversation = self._conversation(messages)
        if not conversation:
            return []
        known = [
            {
                "id": memory.id,
                "title": memory.title,
                "description": memory.description,
                "content": memory.content[:1_000],
                "status": memory.status,
                "validation_count": memory.validation_count,
            }
            for memory in existing
            if memory.type in allowed_types and memory.scope == "project"
        ][:20]
        prompt = f"""The general memory extraction pass missed or failed to parse execution-derived memory.

Extract at most one item of each allowed type: {json.dumps(allowed_types)}. Return [] when there is no durable asset. Return only a JSON array using this schema:
{{"action":"create|merge","target_id":null,"title":"...","description":"...","content":"...","type":"procedure|episode","scope":"project","keywords":["..."],"confidence":0.0,"evidence":"exact reflection quote","supersedes":null}}

For a procedure, state reusable steps, applicability conditions, and the final verification criterion. Do not turn shell workarounds or security-policy bypasses into general procedures. For an episode, record a concrete evidence-backed failure, root cause, and useful remedy; do not create one for routine command noise. Evidence must be copied exactly from reflection.evidence or reflection.verification. Merge into an existing candidate when it describes the same asset.

Existing project execution assets:
{json.dumps(known, ensure_ascii=False)}

Execution reflection:
{json.dumps(reflection.model_dump(), ensure_ascii=False)}

Conversation:
{conversation}
"""
        request = [
            {"role": "system", "content": "You extract conservative, verified execution memory or none."},
            {"role": "user", "content": prompt},
        ]
        result = self._request(request, messages, reflection, evidence_source)
        self.last_fallback_succeeded = result is not None
        if result is None:
            return []
        selected: list[ExtractedMemory] = []
        for item in result:
            if (
                item.type in allowed_types
                and item.scope == "project"
                and item.action in ("create", "merge")
                and not any(existing_item.type == item.type for existing_item in selected)
            ):
                selected.append(item)
        return selected

    def _request(
        self,
        request: list[dict],
        messages: list[dict],
        reflection: SessionReflection | None,
        evidence_source: str,
    ) -> list[ExtractedMemory] | None:
        self.last_error = None
        last_error: Exception | None = None
        for attempt in range(2):
            raw_output = ""
            try:
                response = self.llm.chat(messages=request, tools=None, on_token=None)
                raw_output = response.content
                extracted = self._parse(raw_output)
                filtered = self._filter(extracted, messages, reflection, evidence_source)
                return filtered
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
                                    "Every non-ignore item must include an exact evidence quote from the allowed source."
                                ),
                            },
                        ]
                    )
            except Exception as exc:  # noqa: BLE001 - provider SDKs expose unrelated exception types
                logger.warning("Memory extraction request failed: %s", exc)
                self.last_error = f"{type(exc).__name__}: {exc}"[:500]
                return None
        logger.warning("Memory extraction failed after one repair attempt: %s", last_error)
        self.last_error = f"{type(last_error).__name__}: {last_error}"[:500]
        return None

    @staticmethod
    def supports_procedure(reflection: SessionReflection | None) -> bool:
        return bool(
            reflection
            and reflection.outcome == "success"
            and reflection.verification
            and reflection.tool_executions >= 1
            and reflection.successful_tools >= 1
            and (reflection.evidence or reflection.verification)
        )

    @staticmethod
    def supports_episode(reflection: SessionReflection | None) -> bool:
        return bool(
            reflection
            and reflection.outcome != "unknown"
            and reflection.tool_executions >= 1
            and reflection.failed_tools >= 1
            and (reflection.failures or reflection.root_causes)
            and reflection.evidence
        )

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
    def _filter(
        extracted: list[ExtractedMemory],
        messages: list[dict],
        reflection: SessionReflection | None = None,
        evidence_source: str = "",
    ) -> list[ExtractedMemory]:
        user_text = "\n".join(
            str(message.get("content", ""))
            for message in messages
            if message.get("role") == "user" and isinstance(message.get("content"), str)
        )
        normalized_users = " ".join(user_text.casefold().split())
        normalized_evidence = " ".join(f"{user_text}\n{evidence_source}".casefold().split())
        reflection_quotes = [] if reflection is None else [*reflection.evidence, *reflection.verification]
        normalized_reflection_evidence = " ".join("\n".join(reflection_quotes).casefold().split())
        safe: list[ExtractedMemory] = []
        for item in extracted[:10]:
            if item.action == "ignore":
                continue
            if item.action == "archive" and not item.target_id:
                continue
            combined = " ".join([item.title, item.description, item.content, *item.keywords])
            if _SECRET_RE.search(combined):
                continue
            evidence = " ".join(item.evidence.strip(" \"'“”‘’").casefold().split())
            if not evidence or evidence not in normalized_evidence:
                continue
            if item.type in ("user", "feedback", "profile", "project", "reference") and evidence not in normalized_users:
                continue
            if (
                item.type in ("user", "feedback", "profile", "project", "reference")
                and not MemoryExtractor.is_durable_evidence(item.type, evidence)
            ):
                continue
            if (
                item.type == "procedure"
                and (
                    not MemoryExtractor.supports_procedure(reflection)
                    or evidence not in normalized_reflection_evidence
                    or item.scope != "project"
                )
            ):
                continue
            if (
                item.type == "episode"
                and (
                    not MemoryExtractor.supports_episode(reflection)
                    or evidence not in normalized_reflection_evidence
                    or item.scope != "project"
                )
            ):
                continue
            has_durable_marker = any(marker in evidence for marker in _DURABLE_MARKERS)
            has_local_marker = any(marker in evidence for marker in _LOCAL_MARKERS)
            if item.type not in ("procedure", "episode") and has_local_marker and not has_durable_marker:
                continue
            safe.append(item)
        return safe

    @staticmethod
    def _has_explicit_durable_intent(evidence: str) -> bool:
        return any(marker in evidence for marker in _EXPLICIT_DURABLE_MARKERS)

    @staticmethod
    def _is_user_preference(evidence: str) -> bool:
        return any(marker in evidence for marker in _USER_PREFERENCE_MARKERS) or bool(
            _FUTURE_PREFERENCE_RE.search(evidence)
        )

    @staticmethod
    def _is_profile_fact(evidence: str) -> bool:
        return MemoryExtractor._is_user_preference(evidence) or bool(_PROFILE_RE.search(evidence))

    @staticmethod
    def _is_behavior_feedback(evidence: str) -> bool:
        return MemoryExtractor._is_user_preference(evidence) or bool(_FEEDBACK_RE.search(evidence))

    @staticmethod
    def _is_task_request(evidence: str) -> bool:
        return bool(_TASK_REQUEST_RE.search(evidence))

    @staticmethod
    def is_durable_evidence(memory_type: str, evidence: str) -> bool:
        """Validate evidence for non-execution memories, including legacy retrieval."""
        normalized = " ".join(evidence.casefold().split())
        if not normalized:
            return False
        if memory_type == "user":
            return MemoryExtractor._is_user_preference(normalized)
        if memory_type == "profile":
            return MemoryExtractor._is_profile_fact(normalized)
        if memory_type == "feedback":
            return MemoryExtractor._is_behavior_feedback(normalized)
        if memory_type in ("project", "reference"):
            return not MemoryExtractor._is_task_request(normalized) or (
                MemoryExtractor._has_explicit_durable_intent(normalized)
            )
        return True
