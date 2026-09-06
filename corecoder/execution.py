"""Conservative checks shared by execution and memory admission."""

import json
import re
from pathlib import Path


def incomplete_answer(content: str) -> bool:
    """Reject handoff summaries and explicit unfinished/error terminal answers."""
    text = content.strip()
    return not text or bool(re.search(
        r"(?im)^\s*(?:#+\s*)?(?:conversation summary|current task state|"
        r"会话摘要|对话摘要|上下文摘要)\s*$|"
        r"(?:final (?:audit )?report|final answer|deliverable).{0,100}"
        r"(?:not yet|pending|not been|not written)|"
        r"(?:最终报告|最终审计报告|最终答案).{0,30}(?:尚未|未完成|待完成)|"
        r"^\s*(?:Error:|\[interrupted\]|\(reached maximum tool-call rounds)",
        text,
    ))


def terminal_failure(messages: list[dict]) -> str | None:
    """Check every supplied turn; a later success cannot validate an earlier failure."""
    from .tools.sensitive import sensitive_path

    turns: list[list[dict]] = []
    for message in messages:
        if message.get("role") == "user" or not turns:
            turns.append([])
        turns[-1].append(message)
    for turn in turns:
        if not turn:
            continue
        final = turn[-1]
        if final.get("role") != "assistant" or final.get("tool_calls"):
            return "no terminal assistant answer"
        if incomplete_answer(str(final.get("content") or "")):
            return "unfinished deliverable or handoff summary"
        for message in turn:
            facts = message.get("_execution", {})
            if facts and (
                facts.get("status") != "completed" or facts.get("policy_violations")
            ):
                return "execution did not complete within policy"
            if message.get("role") == "tool" and str(message.get("content", "")).startswith("[Security]"):
                return "security boundary was reached"
            for call in message.get("tool_calls", []):
                function = call.get("function", {})
                if function.get("name") not in {"read_file", "grep"}:
                    continue
                try:
                    args = function.get("arguments", {})
                    args = json.loads(args) if isinstance(args, str) else args
                    path = args.get("file_path") or args.get("path")
                    if path and sensitive_path(Path(path)):
                        return "credential-file access cannot validate a procedure"
                except (ValueError, TypeError, OSError):
                    return "unverifiable read arguments"
    return None
