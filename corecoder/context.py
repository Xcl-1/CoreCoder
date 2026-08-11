"""Multi-layer context compression — v1.0 tiered strategy.

Three layers, progressively more aggressive:

  Layer 1 (tool_snip)  — tool-type-aware truncation (grep: keep all,
                          bash: head+tail, others: first/last lines)
  Layer 2 (summarize)   — incremental LLM summarisation: only summarise
                          new turns since the last checkpoint, merge with
                          existing summary.  O(n²) → O(n) cost.
  Layer 2.5 (layered)   — structured retention: system prompt / user
                          instructions always kept; tool output details
                          compressed to one-line records.
  Layer 3 (hard_collapse) — last resort: drop everything except summary
                          + the most recent turns.

Token counting tries ``tiktoken`` when available, falls back to a
chars/3.5 heuristic that's more accurate than the old //3 estimator.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .llm import LLM

logger = logging.getLogger(__name__)

# ---- optional tiktoken support ------------------------------------------

_TIKTOKEN_ENC = None


def _get_tiktoken():
    """Lazy-load a tiktoken encoder. Returns None if unavailable."""
    global _TIKTOKEN_ENC
    if _TIKTOKEN_ENC is False:  # tried and failed
        return None
    if _TIKTOKEN_ENC is not None:
        return _TIKTOKEN_ENC
    try:
        import tiktoken
        _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
    except Exception:  # noqa: BLE001 — tiktoken may fail in many ways; graceful fallback
        _TIKTOKEN_ENC = False
        return None
    return _TIKTOKEN_ENC


# ---- token estimation ---------------------------------------------------

def _approx_tokens(text: str) -> int:
    """Token count. Uses tiktoken if installed, else chars/3.5."""
    enc = _get_tiktoken()
    if enc:
        return len(enc.encode(text))
    # 3.5 is a better heuristic for mixed en/zh code than 3.0
    return max(1, len(text) // 3)


def estimate_tokens(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        content = m.get("content") or ""
        if isinstance(content, str):
            total += _approx_tokens(content)
        if m.get("tool_calls"):
            total += _approx_tokens(str(m["tool_calls"]))
    return total


# ---- ContextManager -----------------------------------------------------

# how many recent messages to always preserve (never summarise away)
_MIN_KEEP_RECENT = 6


class ContextManager:
    def __init__(self, max_tokens: int = 128_000):
        self.max_tokens = max_tokens
        self._snip_at = int(max_tokens * 0.50)      # 50% → snip
        self._summarize_at = int(max_tokens * 0.70)  # 70% → summarise
        self._collapse_at = int(max_tokens * 0.90)   # 90% → hard collapse

        # incremental summarisation state
        self._last_summary_index: int = 0   # messages before this were already summarised
        self._summary_text: str = ""        # the accumulated summary so far

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def maybe_compress(self, messages: list[dict], llm: LLM | None = None) -> bool:
        """Apply compression layers as needed. Returns True if anything happened."""
        current = estimate_tokens(messages)
        compressed = False

        # Layer 1: tool-type-aware snip
        if current > self._snip_at and self._snip_tool_outputs(messages):
            compressed = True
            current = estimate_tokens(messages)

        # Layer 2: incremental summarisation
        if (current > self._summarize_at and len(messages) > 10
                and self._incremental_summarize(messages, llm, keep_recent=_MIN_KEEP_RECENT)):
            compressed = True
            current = estimate_tokens(messages)

        # Layer 2.5: structured retention — demote old tool details
        if (current > self._summarize_at and len(messages) > 10
                and self._layered_compress(messages, keep_recent=_MIN_KEEP_RECENT)):
            compressed = True
            current = estimate_tokens(messages)

        # Layer 3: hard collapse — last resort
        if current > self._collapse_at and len(messages) > 4:
            self._hard_collapse(messages, llm)
            compressed = True

        return compressed

    # ------------------------------------------------------------------
    # Layer 1 — tool-type-aware snipping
    # ------------------------------------------------------------------

    TOOL_SNIPPERS = {
        # grep returns matched lines — keep them all (the count itself is the value)
        "grep":  None,  # None = never snip
        # bash output — generous head+tail so stderr context stays visible
        "bash": (40, 40),
    }
    # fallback for every other tool
    _DEFAULT_SNIP = (3, 3)

    @classmethod
    def _snip_tool_outputs(cls, messages: list[dict]) -> bool:
        """Layer 1: type-aware tool output truncation.

        - **grep**: never snipped (the match list *is* the value).
        - **bash**: keep 40 head + 40 tail lines (stderr often near the end).
        - **others**: keep 3 head + 3 tail lines (old behaviour).
        """
        changed = False
        # track tool_call_id → tool_name so we know which tool produced each output
        tool_names: dict[str, str] = {}
        for m in messages:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    tool_names[tc.get("id", "")] = tc.get("function", {}).get("name", "")

        for m in messages:
            if m.get("role") != "tool":
                continue
            content = m.get("content", "")
            if len(content) <= 1500:
                continue
            lines = content.splitlines()
            if len(lines) <= 10:
                continue

            tc_id = m.get("tool_call_id", "")
            tool_name = tool_names.get(tc_id, "")
            heads, tails = cls._DEFAULT_SNIP
            if tool_name in cls.TOOL_SNIPPERS:
                spec = cls.TOOL_SNIPPERS[tool_name]
                if spec is None:   # never snip
                    continue
                heads, tails = spec

            if len(lines) <= heads + tails + 2:
                continue

            snipped = (
                "\n".join(lines[:heads])
                + f"\n... ({len(lines)} lines, snipped) ...\n"
                + "\n".join(lines[-tails:])
            )
            m["content"] = snipped
            changed = True
        return changed

    # ------------------------------------------------------------------
    # Layer 2 — incremental summarisation
    # ------------------------------------------------------------------

    def _incremental_summarize(self, messages: list[dict], llm: LLM | None,
                                keep_recent: int = 6) -> bool:
        """Layer 2: only summarise new turns since the last checkpoint.

        Full re-summarisation costs O(n²) — every round we re-summarise
        increasingly large history.  This keeps an accumulated summary and
        only asks the LLM to merge the new part into it.
        """
        split = self._safe_split(messages, keep_recent)
        if split <= self._last_summary_index:
            return False  # nothing new to summarise

        new_material = messages[self._last_summary_index:split]
        tail = messages[split:]

        summary = self._merge_summary(llm, self._summary_text, new_material)
        if not summary:
            return False

        self._summary_text = summary
        self._last_summary_index = 0  # messages will be replaced

        # rebuild: summary block + tail
        messages.clear()
        messages.append({
            "role": "user",
            "content": f"[Conversation summary — incremental]\n{summary}",
        })
        messages.append({
            "role": "assistant",
            "content": "Understood. I have the full context.",
        })
        messages.extend(tail)

        # after rebuild, the summary covers everything before the tail
        self._last_summary_index = 2
        return True

    def _merge_summary(self, llm: LLM | None, existing: str,
                       new_messages: list[dict]) -> str:
        """Ask the LLM to merge new material into an existing summary."""
        flat = self._flatten(new_messages)

        if llm:
            try:
                prompt = _MERGE_PROMPT.format(
                    existing=existing or "(no previous summary)",
                    new_material=flat[:12000],
                )
                resp = llm.chat(
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content.strip()
            except Exception:
                logger.debug("LLM summarisation failed, falling back to regex extraction", exc_info=True)

        # fallback: just prepend the old summary to extracted key info
        extracted = self._extract_key_info(new_messages)
        if existing:
            return f"{existing}\n[new]\n{extracted}"
        return extracted

    # ------------------------------------------------------------------
    # Layer 2.5 — structured layered retention
    # ------------------------------------------------------------------

    @staticmethod
    def _layered_compress(messages: list[dict], keep_recent: int = 6) -> bool:
        """Demote old tool output details while keeping user instructions intact.

        Three tiers:
          - **System / user instructions**: never touched.
          - **Key decision markers** (errors, file writes): one-line record.
          - **Verbose tool output**: truncated to the first meaningful line.
        """
        split = max(0, len(messages) - keep_recent)
        if split <= 2:
            return False

        changed = False
        for i in range(split):
            m = messages[i]
            role = m.get("role", "")

            if role in ("system",):
                continue  # never touch
            if role == "user" and i >= split - 4:
                continue  # keep recent user messages

            content = m.get("content") or ""
            if isinstance(content, str) and len(content) > 800:
                # keep first meaningful line as a record
                first_line = content.split("\n", 1)[0][:200]
                m["content"] = f"[L2.5] {first_line}"
                changed = True

        return changed

    # ------------------------------------------------------------------
    # Layer 3 — hard collapse
    # ------------------------------------------------------------------

    def _hard_collapse(self, messages: list[dict], llm: LLM | None):
        """Layer 3: Emergency compression. Keep only last 4 + summary."""
        split = self._safe_split(messages, 4 if len(messages) > 4 else 2)
        tail = messages[split:]
        summary = self._get_summary(messages[:split], llm)

        messages.clear()
        messages.append({
            "role": "user",
            "content": f"[Hard context reset]\n{summary}",
        })
        messages.append({
            "role": "assistant",
            "content": "Context restored. Continuing from where we left off.",
        })
        messages.extend(tail)

        # reset incremental state since we nuked everything
        self._last_summary_index = 0
        self._summary_text = ""

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _get_summary(self, messages: list[dict], llm: LLM | None) -> str:
        """Full summary (used by hard collapse as a one-shot)."""
        flat = self._flatten(messages)

        if llm:
            try:
                resp = llm.chat(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Compress this conversation into a brief summary. "
                                "Preserve: file paths edited, key decisions made, "
                                "errors encountered, current task state. "
                                "Drop: verbose command output, code listings, "
                                "redundant back-and-forth."
                            ),
                        },
                        {"role": "user", "content": flat[:15000]},
                    ],
                )
                return resp.content
            except Exception:
                logger.debug("Hard collapse summarisation failed, falling back to regex extraction", exc_info=True)

        return self._extract_key_info(messages)

    @staticmethod
    def _safe_split(messages: list[dict], keep_recent: int) -> int:
        """Index where the kept tail should start.

        Walks the boundary back so a 'tool' result is never separated from the
        assistant message whose tool_calls produced it.
        """
        split = max(0, len(messages) - keep_recent)
        while split > 0 and messages[split].get("role") == "tool":
            split -= 1
        return split

    @staticmethod
    def _flatten(messages: list[dict]) -> str:
        parts = []
        for m in messages:
            role = m.get("role", "?")
            text = m.get("content", "") or ""
            if text:
                parts.append(f"[{role}] {text[:400]}")
        return "\n".join(parts)

    @staticmethod
    def _extract_key_info(messages: list[dict]) -> str:
        """Fallback: extract file paths, errors, and decisions without LLM."""
        files_seen: set[str] = set()
        errors: list[str] = []

        for m in messages:
            text = m.get("content", "") or ""
            for match in re.finditer(r'[\w./\-]+\.\w{1,5}', text):
                files_seen.add(match.group())
            for line in text.splitlines():
                if "error" in line.lower():
                    errors.append(line.strip()[:150])

        parts: list[str] = []
        if files_seen:
            parts.append(f"Files touched: {', '.join(sorted(files_seen)[:20])}")
        if errors:
            parts.append(f"Errors seen: {'; '.join(errors[:5])}")
        return "\n".join(parts) or "(no extractable context)"


# ---- prompt template ---------------------------------------------------

_MERGE_PROMPT = """\
You are a conversation compressor. Merge the new conversation segment into the existing summary.

Rules:
- Keep this list of facts current: files touched, key decisions, errors, current task.
- Drop ALL verbose command output and code listings.
- Output ONLY the merged summary text (no JSON, no markdown, no preamble).

Existing summary:
{existing}

New conversation segment:
{new_material}

Merged summary:"""
