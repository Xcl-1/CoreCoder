"""Core agent loop.

This is the heart of CoreCoder.  The pattern is simple:

    user message -> LLM (with tools) -> tool calls? -> execute -> loop
                                      -> text reply? -> return to user

It keeps looping until the LLM responds with plain text (no tool calls),
which means it's done working and ready to report back.

v1.0 adds a role system for multi-agent delegation:
  - **planner**: breaks tasks into steps
  - **executor**: carries out a single step
  - **reviewer**: checks executor output for correctness
  - **researcher**: explores codebase and reports findings
"""

from __future__ import annotations

import asyncio
import enum
import inspect
import logging
import re
import time
import uuid
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .context import ContextManager, estimate_request_tokens, estimate_tokens
from .context_artifacts import ContextArtifactStore
from .execution import incomplete_answer
from .llm import LLM
from .models import PlanRecord, StepRecord, ToolExecRecord
from .prompt import system_prompt
from .replay import ReplayLogger
from .tools import ALL_TOOLS
from .tools.agent import AgentTool
from .tools.base import Tool
from .tools.changes import ChangeTracker, bind_change_tracker, reset_change_tracker
from .tools.retrieve_context import RetrieveContextTool

if TYPE_CHECKING:
    from .memory import MemoryEngine, MemoryWorker
    from .security import Guard
    from .skills import RouteResult, RoutingContext, SkillManager

logger = logging.getLogger(__name__)

_FINALIZATION_EVIDENCE_CHARS = 24_000
_FINALIZATION_SYSTEM_PROMPT = (
    "You are a final-answer formatter, not an investigator. Use only the supplied "
    "user request and tool evidence. Do not inspect further, search for additional "
    "findings, compare more alternatives, or continue open-ended analysis. Treat any "
    "requested count as a maximum, not a quota. If the evidence is insufficient, say "
    "so. Obey the requested output format and length. Never reveal chain-of-thought."
)

_READ_TOOLS = {"read_file", "grep", "glob"}
_SCOPE_PATTERNS = (
    re.compile(r"(?:检索|搜索|读取|检查)?范围(?:限制|限定)(?:在|为|至)?\s*([^，。；;\n]+)"),
    re.compile(
        r"(?:limit|restrict)\s+(?:the\s+)?(?:search|read|inspection)\s+scope\s+to\s+([^.;\n]+)",
        re.IGNORECASE,
    ),
)
_ONLY_TOOL_PATTERNS = (
    re.compile(r"(?:仅|只)允许(?:使用|调用)?\s*([^。；;\n]+)"),
    re.compile(r"(?:only\s+(?:use|allow))\s+([^.;\n]+)", re.IGNORECASE),
)
_FORBIDDEN_TOOL_PATTERNS = (
    re.compile(r"(?:禁止|不得)(?:使用|调用)?\s*([^。；;\n]+)"),
    re.compile(r"(?:do\s+not\s+use|forbid(?:den)?)\s+([^.;\n]+)", re.IGNORECASE),
)


# ---- role system --------------------------------------------------------


class AgentRole(enum.Enum):
    PLANNER = "planner"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"
    RESEARCHER = "researcher"


_ROLE_PROMPTS = {
    AgentRole.PLANNER: (
        "You are a planning agent. Break the task into 3-6 concrete, "
        "verifiable steps. Output ONLY a JSON plan — do not execute anything."
    ),
    AgentRole.EXECUTOR: (
        "You are an executor. Carry out exactly the step given to you. "
        "Report success or failure concisely. Do NOT plan or explore — just execute."
    ),
    AgentRole.REVIEWER: (
        "You are a code reviewer. Examine the changes made by the executor. "
        "Check for: correctness, style consistency, missing edge cases, "
        "potential bugs. Report 'PASS' or list specific issues."
    ),
    AgentRole.RESEARCHER: (
        "You are a research agent. Explore the codebase to answer a specific "
        "question. Use grep, glob, and read_file to gather information. "
        "Report findings concisely — do NOT edit any files."
    ),
}


def role_prompt(role: AgentRole) -> str:
    """Return the role-specific system prompt fragment."""
    return _ROLE_PROMPTS.get(role, "")


def role_tools(role: AgentRole, all_tools: list[Tool]) -> list[Tool]:
    """Filter tools based on role. Reviewer and researcher are read-only."""
    if role in (AgentRole.REVIEWER, AgentRole.RESEARCHER):
        return [t for t in all_tools if t.name in ("read_file", "grep", "glob", "retrieve_context")]
    if role == AgentRole.PLANNER:
        return []  # planner uses no tools — it just thinks
    return all_tools  # executor gets full access


# ---- Agent --------------------------------------------------------------


class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: list[Tool] | None = None,
        max_context_tokens: int = 128_000,
        max_rounds: int = 50,
        replay: bool = True,
        guard: Guard | None = None,
        memory: MemoryEngine | None = None,
        memory_worker: MemoryWorker | None = None,
        skills: SkillManager | None = None,
        changes: ChangeTracker | None = None,
        session_id: str | None = None,
        artifact_store: ContextArtifactStore | None = None,
        context_artifacts_enabled: bool = True,
        context_artifacts_dir: str | Path | None = None,
        context_artifact_threshold: int = 12_000,
        context_artifact_ttl_days: int = 30,
        context_artifact_max_mb: int = 256,
    ):
        self.llm = llm
        self.session_id = session_id or self._new_session_id()
        self.context_artifacts = artifact_store
        if tools is None and self.context_artifacts is None and context_artifacts_enabled:
            self.context_artifacts = ContextArtifactStore(
                self.session_id,
                root=context_artifacts_dir,
                threshold_chars=context_artifact_threshold,
                ttl_seconds=context_artifact_ttl_days * 24 * 60 * 60,
                max_total_bytes=context_artifact_max_mb * 1024 * 1024,
            )
        self.tools = list(tools if tools is not None else ALL_TOOLS)
        if self.context_artifacts is not None and not any(
            tool.name == "retrieve_context" for tool in self.tools
        ):
            self.tools.append(RetrieveContextTool(self.context_artifacts))
        self._tool_by_name = {t.name: t for t in self.tools}
        self.messages: list[dict] = []
        self._turn_messages: list[dict] = []
        self._policy_violations = 0
        self.context = ContextManager(
            max_tokens=max_context_tokens,
            artifact_store=self.context_artifacts,
        )
        self.max_rounds = max_rounds
        self._system = system_prompt(self.tools)
        self._step_number = 0
        self.guard = guard
        self.memory = memory
        self.memory_worker = memory_worker
        self._memory_prompt = ""
        self._memory_context_loaded = False
        self._memory_finalized = False
        self._memory_checkpoint_turn_id: str | None = None
        self.skills = skills
        self._skill_prompt = ""
        self._skill_forbidden_tools: set[str] = set()
        self._turn_allowed_tools: set[str] | None = None
        self._turn_forbidden_tools: set[str] = set()
        self._turn_read_scope: tuple[Path, ...] = ()
        self._active_skill_risk = "low"
        self._active_skill_ids: list[str] = []
        self._skill_tool_successes = 0
        self._skill_tool_failures = 0
        self._skill_outcome_recorded = False
        self.changes = changes or ChangeTracker()
        # replay log — on by default in production, off in tests
        self._replay = ReplayLogger(self.session_id) if replay else None
        if self._replay:
            self._replay.open()

        # wire up sub-agent capability
        for t in self.tools:
            if isinstance(t, AgentTool):
                t._parent_agent = self

    def _full_messages(self) -> list[dict]:
        system = self._system
        if self._memory_prompt:
            system = f"{system}\n\n{self._memory_prompt}"
        if self._skill_prompt:
            system = f"{system}\n\n{self._skill_prompt}"
        return [{"role": "system", "content": system}] + [
            {k: v for k, v in message.items() if not k.startswith("_")}
            for message in self.messages
        ]

    def _append_message(self, message: dict) -> None:
        self.messages.append(message)
        self._turn_messages.append(deepcopy(message))

    def _tool_schemas(self) -> list[dict]:
        forbidden = self._skill_forbidden_tools | self._turn_forbidden_tools
        schemas = []
        for tool in self.tools:
            if tool.name in forbidden:
                continue
            if self._turn_allowed_tools is not None and tool.name not in self._turn_allowed_tools:
                continue
            schema = tool.schema()
            if tool.name in _READ_TOOLS and self._turn_read_scope:
                allowed = ", ".join(str(path) for path in self._turn_read_scope)
                schema["function"]["description"] += (
                    f" Active user-requested read scope: {allowed}. "
                    "Do not target a parent directory."
                )
            schemas.append(schema)
        return schemas

    def _context_overhead_tokens(self) -> int:
        """Budget system additions, tool schemas, and reserved model output."""
        system_message = self._full_messages()[0]
        configured = getattr(self.llm, "extra", {}).get("max_tokens", 4096)
        try:
            output_reserve = int(configured)
        except (TypeError, ValueError):
            output_reserve = 4096
        output_reserve = min(
            max(256, output_reserve),
            max(256, self.context.max_tokens // 4),
        )
        return estimate_request_tokens(
            [system_message],
            tools=self._tool_schemas(),
            reserve_tokens=output_reserve,
        )

    @staticmethod
    def _finalization_messages(full_msgs: list[dict]) -> list[dict]:
        """Build a compact, tool-free transcript for an empty-answer retry.

        Thinking-model reasoning and the normal system/skill prompt are deliberately
        omitted: the retry should format evidence already gathered, not restart the
        task or continue an open-ended review.
        """
        last_user_index = next(
            (
                index
                for index in range(len(full_msgs) - 1, -1, -1)
                if full_msgs[index].get("role") == "user"
            ),
            0,
        )
        current_turn = full_msgs[last_user_index:]
        user_request = next(
            (
                str(message.get("content") or "")
                for message in current_turn
                if message.get("role") == "user"
            ),
            "",
        )
        evidence_blocks = [
            str(message.get("content"))
            for message in current_turn
            if message.get("role") in {"tool", "assistant"}
            and message.get("content")
        ]
        evidence = "\n\n".join(evidence_blocks)
        if len(evidence) > _FINALIZATION_EVIDENCE_CHARS:
            half = (_FINALIZATION_EVIDENCE_CHARS - 64) // 2
            evidence = (
                evidence[:half].rstrip()
                + "\n\n[tool evidence truncated for finalization]\n\n"
                + evidence[-half:].lstrip()
            )
        prompt = (
            f"Original user request:\n{user_request}\n\n"
            f"Tool evidence already gathered:\n{evidence or '[none]'}\n\n"
            "Return the final answer now. Do not call tools and do not perform "
            "additional investigation or analysis."
        )
        return [
            {"role": "system", "content": _FINALIZATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

    async def chat(self, user_input: str,
                   on_token: Callable[[str], None] | None = None,
                   on_tool: Callable[[str, dict[str, Any]], None] | None = None,
                   routing_context: RoutingContext | dict | None = None) -> str:
        self._turn_messages = []
        self._policy_violations = 0
        status = "failed"
        try:
            answer = await self._chat(user_input, on_token, on_tool, routing_context)
            status = "partial" if incomplete_answer(answer) or self._policy_violations else "completed"
            return answer
        except (Exception, KeyboardInterrupt, asyncio.CancelledError) as exc:
            answered = {m.get("tool_call_id") for m in self._turn_messages if m.get("role") == "tool"}
            pending = [call for m in self._turn_messages for call in m.get("tool_calls", [])
                       if call.get("id") not in answered]
            for call in pending:
                self._append_message({"role": "tool", "tool_call_id": call["id"], "content": "[interrupted]"})
            self._append_message({
                "role": "assistant",
                "content": f"Error: execution interrupted ({type(exc).__name__}); task not completed.",
            })
            self._record_skill_outcome("failure")
            raise
        finally:
            if self._turn_messages:
                self._turn_messages[-1]["_execution"] = {
                    "status": status, "policy_violations": self._policy_violations,
                }
                if self.messages:
                    self.messages[-1]["_execution"] = dict(self._turn_messages[-1]["_execution"])

    async def _chat(self, user_input: str,
                   on_token: Callable[[str], None] | None = None,
                   on_tool: Callable[[str, dict[str, Any]], None] | None = None,
                   routing_context: RoutingContext | dict | None = None) -> str:
        """Process one user message. May involve multiple LLM/tool rounds."""
        self._load_turn_policy(user_input)
        route_result = self._load_skill_context(user_input, routing_context)
        self._load_memory_context(user_input)
        self._append_message({"role": "user", "content": user_input})
        if route_result is not None and route_result.needs_clarification:
            answer = route_result.clarification
            self._append_message({"role": "assistant", "content": answer})
            return answer
        self.context.request_overhead_tokens = self._context_overhead_tokens()
        await asyncio.to_thread(self.context.maybe_compress, self.messages, self.llm)

        for _ in range(self.max_rounds):
            self._step_number += 1
            step_start = time.monotonic()
            full_msgs = self._full_messages()
            tool_schemas = self._tool_schemas()
            est_tokens = estimate_request_tokens(full_msgs, tools=tool_schemas)

            resp = await asyncio.to_thread(
                self.llm.chat,
                messages=full_msgs,
                tools=tool_schemas,
                on_token=on_token,
            )

            # no tool calls -> LLM is done, log the final step and return
            if not resp.tool_calls:
                if incomplete_answer(resp.content) or resp.finish_reason in {"length", "content_filter"}:
                    # Thinking models can exhaust their output budget in
                    # reasoning_content before emitting a user-visible answer.
                    # Make one tool-free attempt to turn the gathered evidence
                    # into a concise final response instead of silently
                    # returning an empty string.
                    self._log_step(self._step_number, len(full_msgs), est_tokens,
                                   resp, [], step_start)
                    recovery_messages = self._finalization_messages(self._turn_messages)
                    self._step_number += 1
                    recovery_start = time.monotonic()
                    recovery = await asyncio.to_thread(
                        self.llm.chat,
                        messages=recovery_messages,
                        tools=None,
                        on_token=on_token,
                    )
                    if recovery.tool_calls:
                        # No tools were offered for finalization. Never persist
                        # hallucinated calls without matching tool replies.
                        recovery = recovery.model_copy(update={"tool_calls": [], "content": ""})
                    if incomplete_answer(recovery.content) or recovery.finish_reason in {"length", "content_filter"}:
                        reason = recovery.finish_reason or resp.finish_reason or "unknown"
                        recovery = recovery.model_copy(update={
                            "content": (
                                "Error: the model produced no final answer "
                                f"(finish_reason={reason}). Try a smaller task or a larger output limit."
                            ),
                            "reasoning_content": "",
                        })
                    self._append_message(recovery.message)
                    self._log_step(
                        self._step_number,
                        len(recovery_messages),
                        estimate_tokens(recovery_messages),
                        recovery,
                        [],
                        recovery_start,
                    )
                    self._record_skill_outcome(
                        "failure" if recovery.content.startswith("Error:") else self._skill_outcome()
                    )
                    return recovery.content
                self._append_message(resp.message)
                self._log_step(self._step_number, len(full_msgs), est_tokens,
                               resp, [], step_start)
                self._record_skill_outcome(self._skill_outcome())
                return resp.content

            # tool calls -> execute (async gather for parallelism)
            self._append_message(resp.message)

            try:
                results = await self._exec_tools_async(resp.tool_calls, on_tool)
                for tc, (result, _elapsed, _success) in results:
                    if _success:
                        self._skill_tool_successes += 1
                    else:
                        self._skill_tool_failures += 1
                    prepared_result = self.context.prepare_tool_result(result, tc.name)
                    self._append_message({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": prepared_result,
                    })
            except KeyboardInterrupt:
                # Ctrl+C mid-execution would leave the assistant tool_calls
                # message without replies, poisoning the next request; backfill
                self._answer_pending_tool_calls(resp.tool_calls)
                raise

            # log the completed step
            self._log_step(self._step_number, len(full_msgs), est_tokens,
                           resp, results, step_start)

            # compress if tool outputs are big
            self.context.request_overhead_tokens = self._context_overhead_tokens()
            await asyncio.to_thread(self.context.maybe_compress, self.messages, self.llm)

        self._record_skill_outcome("failure")
        answer = "(reached maximum tool-call rounds)"
        self._append_message({"role": "assistant", "content": answer})
        return answer

    async def _exec_tool(self, tc: Any) -> tuple[str, float, bool]:
        """Execute a single tool call. Returns (result, elapsed_ms, success)."""
        tool = self._tool_by_name.get(tc.name)
        if tool is None:
            logger.warning("Unknown tool requested: %s", tc.name)
            return f"Error: unknown tool '{tc.name}'", 0, False
        if tc.name in self._skill_forbidden_tools:
            self._policy_violations += 1
            return f"Error: active skill policy forbids tool '{tc.name}'", 0, False
        if (
            tc.name in self._turn_forbidden_tools
            or (self._turn_allowed_tools is not None and tc.name not in self._turn_allowed_tools)
        ):
            self._policy_violations += 1
            return f"[Security] Blocked: the user-requested tool policy forbids '{tc.name}'", 0, False
        # validate arguments first so a TypeError raised *inside* the tool isn't
        # mislabelled as a bad-arguments error from the caller
        properties = set(tool.parameters.get("properties", {}))
        required = set(tool.parameters.get("required", ()))
        unknown = set(tc.arguments) - properties
        missing = required - set(tc.arguments)
        argument_errors: list[str] = []
        if unknown:
            argument_errors.append(f"unexpected: {', '.join(sorted(unknown))}")
        if missing:
            argument_errors.append(f"missing: {', '.join(sorted(missing))}")
        if argument_errors:
            expected = ", ".join(sorted(properties)) or "no arguments"
            message = (
                f"Error: bad arguments for {tc.name}: {'; '.join(argument_errors)}; "
                f"expected: {expected}"
            )
            return (
                message,
                0,
                False,
            )
        validation_target = (
            tool._execute_sync if type(tool).execute is Tool.execute else tool.execute
        )
        try:
            inspect.signature(validation_target).bind(**tc.arguments)
        except TypeError as e:
            logger.debug("Bad arguments for %s: %s", tc.name, e)
            return f"Error: bad arguments for {tc.name}: {e}", 0, False

        scope_error = self._read_scope_error(tc.name, tc.arguments)
        constrained_targets = self._constrained_read_targets(tc.name, tc.arguments) if scope_error else ()
        if scope_error and not constrained_targets:
            self._policy_violations += 1
            return f"[Security] Blocked: {scope_error}", 0, False

        # ---- security review ----
        security_confirmed = False
        if self.guard is not None:
            decision = self.guard.review(tc.name, tc.arguments)
            if not decision.allowed:
                self._policy_violations += 1
                return f"[Security] Blocked: {decision.reason}", 0, False
            security_confirmed = decision.user_confirmed
        if (
            self._active_skill_risk == "high"
            and tool.side_effect != "none"
            and not security_confirmed
        ):
            reason = (
                "The active skill is high risk and this tool may change state; "
                "explicit confirmation is required before execution"
            )
            if self.guard is None:
                return f"[Security] Blocked: {reason}", 0, False
            confirmation = self.guard.request_confirmation(
                tc.name,
                tc.arguments,
                reason,
                source="skill-risk",
            )
            if not confirmation.allowed:
                return f"[Security] Blocked: {confirmation.reason}", 0, False

        t0 = time.monotonic()
        tracker_token = bind_change_tracker(self.changes)
        try:
            if constrained_targets:
                chunks = []
                for target in constrained_targets:
                    scoped_arguments = dict(tc.arguments)
                    scoped_arguments["path"] = str(target)
                    scoped_result = await tool.execute(**scoped_arguments)
                    chunks.append(f"[Scope: {target}]\n{scoped_result}")
                result = "[Scope] Parent search constrained to user-approved roots.\n" + "\n".join(chunks)
            else:
                result = await tool.execute(**tc.arguments)
            if result.startswith("[Security]"):
                self._policy_violations += 1
            # ---- output sanitisation ----
            if self.guard is not None:
                result = self.guard.sanitize(result)
            elapsed = (time.monotonic() - t0) * 1000
            success = not result.startswith(("Error", "[Security]"))
            if not success:
                logger.debug("Tool %s failed: %s", tc.name, result[:200])
            return result, elapsed, success
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            logger.exception("Tool %s raised exception", tc.name)
            return f"Error executing {tc.name}: {e}", elapsed, False
        finally:
            reset_change_tracker(tracker_token)

    async def _exec_tools_async(self, tool_calls: list[Any],
                                 on_tool: Callable[[str, dict[str, Any]], None] | None = None
                                 ) -> list[tuple[Any, tuple[str, float, bool]]]:
        """Run tool calls with read-priority scheduling.

        Strategy:
        1. All read-only tools (read_file, grep, glob) start immediately in
           parallel — they never conflict with each other.
        2. Write tools (write_file, edit_file, edit_ast) are grouped by
           target file path.  Within each group, if there was a preceding
           read for the same file, the read completes first.
        3. Everything else (bash, agent) runs in parallel alongside reads.

        This gives the same wall-clock as blind ``asyncio.gather`` for
        independent calls, but prevents race conditions when the LLM issues
        a read+edit pair for the same file in one round.
        """
        for tc in tool_calls:
            if on_tool:
                on_tool(tc.name, tc.arguments)

        # classify
        readers: list = []   # (tc,) — safe to run fully parallel
        writers: list = []   # (tc,) — grouped by target path below
        others: list = []    # (tc,) — bash, agent, etc.

        for tc in tool_calls:
            name = tc.name
            if name in ("read_file", "grep", "glob", "retrieve_context"):
                readers.append(tc)
            elif name in ("write_file", "edit_file", "edit_ast"):
                writers.append(tc)
            else:
                others.append(tc)

        # build tasks: readers + others all start in parallel
        tasks: dict[str, asyncio.Task] = {}  # tc.id → task

        def _launch(tc):
            task = asyncio.create_task(self._exec_tool(tc))
            tasks[tc.id] = task
            return tc, task

        launched = []
        for tc in readers + others:
            launched.append(_launch(tc))

        # writers: group by file_path so we serialize reads→writes on the
        # same path when a matching read was already launched
        writer_groups: dict[str, list] = {}
        for tc in writers:
            path = tc.arguments.get("file_path", "") or ""
            writer_groups.setdefault(path, []).append(tc)

        for path, wlist in writer_groups.items():
            # if any reader targeted the same path, wait for those reads first
            for rtc in readers:
                rpath = rtc.arguments.get("file_path", "") or ""
                if rpath == path and rtc.id in tasks:
                    await tasks[rtc.id]  # wait for the read to finish
            for wtc in wlist:
                launched.append(_launch(wtc))

        # gather all remaining tasks
        results = []
        for tc, coro in launched:
            try:
                results.append((tc, await coro))
            except Exception:  # noqa: BLE001 — guard against unexpected internal errors
                # _exec_tool never raises (it catches internally),
                # but guard anyway
                results.append((tc, ("Error: internal error", 0, False)))

        return results

    def _answer_pending_tool_calls(self, tool_calls: list[Any]) -> None:
        """Backfill a tool reply for every call that didn't get one.

        OpenAI-compatible APIs reject a request where an assistant message has
        tool_calls without a matching tool reply for each id, so this keeps the
        history valid when execution is interrupted partway through.
        """
        answered = {m.get("tool_call_id") for m in self.messages if m.get("role") == "tool"}
        for tc in tool_calls:
            if tc.id not in answered:
                self._append_message({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "[interrupted]",
                })

    def reset(self):
        """Clear conversation history."""
        self.messages.clear()
        self._turn_messages.clear()
        self._policy_violations = 0
        max_context_tokens = self.context.max_tokens
        previous_artifact_store = self.context_artifacts
        self._step_number = 0
        self._memory_prompt = ""
        self._memory_context_loaded = False
        self._memory_finalized = False
        self._memory_checkpoint_turn_id = None
        self._skill_prompt = ""
        self._skill_forbidden_tools.clear()
        self._turn_allowed_tools = None
        self._turn_forbidden_tools.clear()
        self._turn_read_scope = ()
        if self.skills is not None:
            self.skills.clear_pins()
        self.session_id = self._new_session_id()
        if previous_artifact_store is not None:
            self.context_artifacts = ContextArtifactStore(
                self.session_id,
                root=previous_artifact_store.root,
                threshold_chars=previous_artifact_store.threshold_chars,
                preview_chars=previous_artifact_store.preview_chars,
                ttl_seconds=previous_artifact_store.ttl_seconds,
                max_total_bytes=previous_artifact_store.max_total_bytes,
            )
            for tool in self.tools:
                if isinstance(tool, RetrieveContextTool):
                    tool.store = self.context_artifacts
        self.context = ContextManager(
            max_tokens=max_context_tokens,
            artifact_store=self.context_artifacts,
        )
        if self._replay:
            self._replay.close()
            self._replay = ReplayLogger(self.session_id)
            self._replay.open()

    def _load_turn_policy(self, user_input: str) -> None:
        """Compile explicit per-turn tool and read-scope constraints from the request."""
        known_tools = set(self._tool_by_name)
        self._turn_allowed_tools = self._tool_names_in_clauses(
            user_input, _ONLY_TOOL_PATTERNS, known_tools
        ) or None
        self._turn_forbidden_tools = self._tool_names_in_clauses(
            user_input, _FORBIDDEN_TOOL_PATTERNS, known_tools
        )
        self._turn_read_scope = self._read_scope_in_request(user_input)

    @staticmethod
    def _tool_names_in_clauses(
        user_input: str,
        patterns: tuple[re.Pattern[str], ...],
        known_tools: set[str],
    ) -> set[str]:
        names: set[str] = set()
        for pattern in patterns:
            for match in pattern.finditer(user_input):
                words = set(re.findall(r"[A-Za-z][A-Za-z0-9_]*", match.group(1)))
                names.update(words & known_tools)
        return names

    @staticmethod
    def _read_scope_in_request(user_input: str) -> tuple[Path, ...]:
        cwd = Path.cwd().resolve()
        roots: list[Path] = []
        for pattern in _SCOPE_PATTERNS:
            match = pattern.search(user_input)
            if match is None:
                continue
            values = re.split(
                r"\s*(?:、|,|，|\band\b|和|以及)\s*",
                match.group(1),
                flags=re.IGNORECASE,
            )
            for value in values:
                candidate = value.strip().strip("`'\" ")
                if not candidate:
                    continue
                path = Path(candidate).expanduser()
                resolved = (path if path.is_absolute() else cwd / path).resolve()
                if resolved.exists() and resolved not in roots:
                    roots.append(resolved)
            break
        return tuple(roots)

    def _read_scope_error(self, tool_name: str, arguments: dict[str, Any]) -> str | None:
        if tool_name not in _READ_TOOLS or not self._turn_read_scope:
            return None
        targets = self._read_scope_targets(tool_name, arguments)
        if all(
            any(target == root or target.is_relative_to(root) for root in self._turn_read_scope)
            for target in targets
        ):
            return None
        allowed = ", ".join(str(path) for path in self._turn_read_scope)
        requested = ", ".join(str(path) for path in targets)
        return f"read target '{requested}' is outside the user-requested scope: {allowed}"

    def _constrained_read_targets(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> tuple[Path, ...]:
        """Safely narrow a parent grep/glob to the explicitly approved roots."""
        if tool_name not in {"grep", "glob"} or not self._turn_read_scope:
            return ()
        raw_path = Path(str(arguments.get("path") or ".")).expanduser()
        requested = (raw_path if raw_path.is_absolute() else Path.cwd() / raw_path).resolve()
        if not any(root.is_relative_to(requested) for root in self._turn_read_scope):
            return ()
        include = str(arguments.get("include") or "")
        targets = []
        for root in self._turn_read_scope:
            if not root.is_relative_to(requested):
                continue
            if tool_name == "glob" and not root.is_dir():
                continue
            if tool_name == "grep" and root.is_file() and include and not root.match(include):
                continue
            targets.append(root)
        return tuple(targets)

    @staticmethod
    def _read_scope_targets(tool_name: str, arguments: dict[str, Any]) -> tuple[Path, ...]:
        raw_path = arguments.get("file_path") if tool_name == "read_file" else arguments.get("path", ".")
        base = Path(str(raw_path or ".")).expanduser()
        if not base.is_absolute():
            base = Path.cwd() / base
        targets = [base]
        if tool_name != "glob":
            return (base.resolve(),)
        for part in Path(str(arguments.get("pattern", ""))).parts:
            if any(marker in part for marker in "*?["):
                break
            alternatives = [part]
            if part.startswith("{") and part.endswith("}"):
                alternatives = [value.strip() for value in part[1:-1].split(",") if value.strip()]
            targets = [target / value for target in targets for value in alternatives]
        return tuple(target.resolve() for target in targets)

    @staticmethod
    def _new_session_id() -> str:
        return f"session_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    def learn(self) -> list:
        """Extract durable memory from this session once, without blocking shutdown."""
        if self.memory is None or self._memory_finalized:
            return []
        self._memory_finalized = True
        try:
            replay_path = self._replay.path if self._replay else None
            if replay_path:
                return self.memory.learn(self._turn_messages or self.messages, self.session_id, replay_path=replay_path)
            return self.memory.learn(self._turn_messages or self.messages, self.session_id)
        except Exception:
            logger.warning("Failed to learn from session", exc_info=True)
            return []

    def _load_memory_context(self, user_input: str) -> None:
        if self.memory is None:
            return
        self._memory_context_loaded = True
        try:
            self._memory_prompt = self.memory.build_prompt(user_input)
        except Exception:
            logger.warning("Failed to retrieve cross-session memory", exc_info=True)
            self._memory_prompt = ""

    def _load_skill_context(
        self,
        user_input: str,
        routing_context: RoutingContext | dict | None = None,
    ) -> RouteResult | None:
        """Route and activate skills for exactly one user turn."""
        self._skill_prompt = ""
        self._skill_forbidden_tools.clear()
        self._active_skill_risk = "low"
        self._active_skill_ids = []
        self._skill_tool_successes = 0
        self._skill_tool_failures = 0
        self._skill_outcome_recorded = False
        if self.skills is None:
            return None
        try:
            result = self.skills.route(
                user_input,
                {tool.name for tool in self.tools},
                context=routing_context,
            )
            self._skill_prompt = result.prompt
            self._skill_forbidden_tools = result.forbidden_tools
            self._active_skill_ids = result.selected_ids
            risk_order = {"low": 0, "medium": 1, "high": 2}
            route_risks = [result.signature.risk]
            route_risks.extend(
                item.skill.manifest.routing.risk for item in result.selected
            )
            if route_risks:
                self._active_skill_risk = max(
                    route_risks,
                    key=risk_order.__getitem__,
                )
            return result
        except Exception:
            logger.warning("Failed to route task skills", exc_info=True)
            return None

    def _skill_outcome(self) -> str:
        if self._skill_tool_failures and not self._skill_tool_successes:
            return "failure"
        if self._skill_tool_failures:
            return "partial"
        return "success"

    def _record_skill_outcome(self, outcome: str) -> None:
        """Feed one terminal result back into persistent routing telemetry."""
        if (
            self._skill_outcome_recorded
            or self.skills is None
            or not self._active_skill_ids
        ):
            return
        self._skill_outcome_recorded = True
        try:
            self.skills.record_outcome(self._active_skill_ids, outcome)
        except Exception:
            logger.warning("Failed to record skill execution outcome", exc_info=True)

    def checkpoint_memory(self) -> None:
        """Durably queue the latest turn, then schedule non-blocking learning."""
        if self.memory is None or not hasattr(self.memory, "checkpoint"):
            return
        try:
            transcript = self._turn_messages or self.messages
            turn_id = self.memory.latest_turn_id(transcript)
            if turn_id is None or turn_id == self._memory_checkpoint_turn_id:
                return
            # Incremental turns already contain tool calls and results, so the
            # worker need not reread the full (and ever-growing) replay file.
            path = self.memory.checkpoint(transcript, self.session_id)
            if path is None:
                return
            self._memory_checkpoint_turn_id = turn_id
            if self.memory_worker is not None:
                self.memory_worker.submit(self.session_id)
        except Exception:
            logger.warning("Failed to checkpoint pending memory", exc_info=True)

    def mark_memory_checkpointed(self) -> None:
        """Mark loaded history as old so resume does not queue it as a new turn."""
        if self.memory is not None:
            self._memory_checkpoint_turn_id = self.memory.latest_turn_id(self.messages)

    def close(self):
        """Stop background intake and close replay without waiting on the network."""
        if self.memory_worker is not None:
            self.memory_worker.close(wait=False)
        if self._replay:
            self._replay.close()

    async def spawn(
        self,
        task: str,
        role: AgentRole = AgentRole.EXECUTOR,
        reviewer: bool = False,
    ) -> str:
        """Spawn a sub-agent with a specific role and return its result.

        The sub-agent gets:
        - A role-specific system prompt
        - Role-filtered tools (reviewer/researcher are read-only)
        - An independent context window
        - Optional reviewer pass after executor completes

        This is the foundation of multi-agent delegation — the parent agent
        can spawn N specialised children for different parts of a task.
        """
        tools = self._tools_for_role(role)

        sub = Agent(
            llm=self.llm,
            tools=tools,
            max_context_tokens=self.context.max_tokens,
            max_rounds=min(self.max_rounds, 15),
            replay=False,  # sub-agents don't write their own replay logs
            guard=self.guard,  # inherit parent's security policy
            changes=self.changes,  # sub-agent edits belong to the parent session
        )

        # inject role-specific prompt as the system message
        role_instruction = role_prompt(role)
        sub._system = f"{sub._system}\n\n[Role: {role.value}]\n{role_instruction}"

        try:
            result = await sub.chat(task)

            # optional reviewer pass
            if reviewer and role == AgentRole.EXECUTOR and result:
                review = await self._review(executor_result=result, task=task)
                result = f"{result}\n\n[Reviewer ({AgentRole.REVIEWER.value})]\n{review}"

            # trim long results
            if len(result) > 5000:
                result = result[:4500] + "\n... (sub-agent output truncated)"
            return result
        except (OSError, ValueError, RuntimeError) as e:
            logger.error("Sub-agent (%s) error: %s", role.value, e)
            return f"Sub-agent ({role.value}) error: {e}"
        finally:
            sub.close()

    def _tools_for_role(self, role: AgentRole) -> list[Tool]:
        """Return role tools after applying the active parent skill policy."""
        return [
            tool for tool in role_tools(role, self.tools)
            if tool.name != "agent" and tool.name not in self._skill_forbidden_tools
        ]

    async def _review(self, executor_result: str, task: str) -> str:
        """Run a lightweight reviewer pass on executor output."""
        review_prompt = (
            f"Task: {task}\n\n"
            f"Executor output:\n{executor_result[:3000]}\n\n"
            f"Review the above. Report PASS or list specific issues."
        )
        tools = self._tools_for_role(AgentRole.REVIEWER)

        reviewer = Agent(
            llm=self.llm,
            tools=tools,
            max_context_tokens=self.context.max_tokens,
            max_rounds=5,
            replay=False,
        )
        reviewer._system = f"{reviewer._system}\n\n[Role: reviewer]\n{_ROLE_PROMPTS[AgentRole.REVIEWER]}"
        try:
            return await reviewer.chat(review_prompt)
        finally:
            reviewer.close()

    async def plan(self, task: str) -> PlanRecord:
        """Generate a structured execution plan for a complex task.

        Returns a PlanRecord with a goal and ordered steps.  The user
        reviews and confirms the plan before the agent executes it.
        """

        prompt = f"""You are a software engineering planner. Given the task below, produce a structured execution plan as JSON.

Return ONLY a JSON object with this exact structure:
{{"goal": "<one-line summary>", "steps": [{{"id": 1, "action": "<what to do>", "tool": "<suggested tool name or empty>", "expected": "<what success looks like>"}}]}}

Rules:
- Break complex tasks into 3-8 concrete steps.
- Each step should be a single, verifiable action.
- Suggest the most appropriate CoreCoder tool for each step (bash, read_file, write_file, edit_file, edit_ast, grep, glob, or empty string).
- Order steps logically — read before edit, test after change.

Task: {task}

Plan (JSON only):"""

        resp = await asyncio.to_thread(
            self.llm.chat,
            messages=[{"role": "user", "content": prompt}],
            tools=None,
            on_token=None,
        )

        # extract JSON from the response (may be wrapped in ```json blocks)
        text = resp.content.strip()
        if "```" in text:
            # extract content between first ```json and last ```
            text = text.split("```json", 1)[-1].split("```", 1)[0].strip()
        elif text.startswith("{"):
            pass  # raw JSON
        else:
            # try to find the first { ... } block
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                text = text[start:end]

        plan = PlanRecord.model_validate_json(text)
        return plan

    def _log_step(self, step: int, msg_count: int, est_tokens: int,
                  resp: Any, results: list[tuple[Any, tuple[str, float, bool]]],
                  step_start: float) -> None:
        """Write one StepRecord to the replay log, if enabled."""
        if not self._replay:
            return

        # build tool execution records (truncate long results)
        execs: list[ToolExecRecord] = []
        for tc, (result, elapsed, success) in results:
            truncated = result[:5000] if len(result) > 5000 else result
            error_msg = None
            if not success:
                error_msg = result[:500]
            execs.append(ToolExecRecord(
                name=tc.name,
                arguments=tc.arguments,
                result=truncated,
                duration_ms=round(elapsed, 2),
                success=success,
                error=error_msg,
            ))

        step_duration = (time.monotonic() - step_start) * 1000
        skill_route: dict[str, Any] = {}
        if self.skills is not None and self.skills.last_result is not None:
            routed = self.skills.last_result
            skill_route = {
                "selected": routed.selected_ids,
                "decision": routed.decision,
                "confidence": routed.confidence,
                "margin": routed.margin,
                "clarification": routed.clarification,
                "signature": routed.signature.model_dump(mode="json"),
                "candidates": [
                    {
                        "id": item.skill.manifest.id,
                        "score": item.score,
                        "recall_score": item.recall_score,
                        "confidence": item.confidence,
                        "shadow": item.shadow,
                        "reasons": item.reasons,
                    }
                    for item in routed.candidates
                ],
                "rejected": routed.rejected,
                "prompt_chars": len(routed.prompt),
            }
        record = StepRecord(
            step=step,
            messages_count=msg_count,
            estimated_input_tokens=est_tokens,
            llm_response=resp,
            tool_executions=execs,
            skill_route=skill_route,
            step_duration_ms=round(step_duration, 2),
        )
        self._replay.log(record)
