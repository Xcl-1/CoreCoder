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
import time
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .context import ContextManager, estimate_tokens
from .llm import LLM
from .models import PlanRecord, StepRecord, ToolExecRecord
from .prompt import system_prompt
from .replay import ReplayLogger
from .tools import ALL_TOOLS
from .tools.agent import AgentTool
from .tools.base import Tool

if TYPE_CHECKING:
    from .memory import MemoryEngine
    from .security import Guard

logger = logging.getLogger(__name__)


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
        return [t for t in all_tools if t.name in ("read_file", "grep", "glob")]
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
        session_id: str | None = None,
    ):
        self.llm = llm
        self.tools = tools if tools is not None else ALL_TOOLS
        self._tool_by_name = {t.name: t for t in self.tools}
        self.messages: list[dict] = []
        self.context = ContextManager(max_tokens=max_context_tokens)
        self.max_rounds = max_rounds
        self._system = system_prompt(self.tools)
        self._step_number = 0
        self.guard = guard
        self.memory = memory
        self._memory_prompt = ""
        self._memory_context_loaded = False
        self._memory_finalized = False

        # replay log — on by default in production, off in tests
        self._replay = ReplayLogger() if replay else None
        if self._replay:
            self._replay.open()
        self.session_id = session_id or self._new_session_id()

        # wire up sub-agent capability
        for t in self.tools:
            if isinstance(t, AgentTool):
                t._parent_agent = self

    def _full_messages(self) -> list[dict]:
        system = self._system
        if self._memory_prompt:
            system = f"{system}\n\n{self._memory_prompt}"
        return [{"role": "system", "content": system}] + self.messages

    def _tool_schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools]

    async def chat(self, user_input: str,
                   on_token: Callable[[str], None] | None = None,
                   on_tool: Callable[[str, dict[str, Any]], None] | None = None) -> str:
        """Process one user message. May involve multiple LLM/tool rounds."""
        self._load_memory_context(user_input)
        self.messages.append({"role": "user", "content": user_input})
        await asyncio.to_thread(self.context.maybe_compress, self.messages, self.llm)

        for _ in range(self.max_rounds):
            self._step_number += 1
            step_start = time.monotonic()
            full_msgs = self._full_messages()
            est_tokens = estimate_tokens(full_msgs)

            resp = await asyncio.to_thread(
                self.llm.chat,
                messages=full_msgs,
                tools=self._tool_schemas(),
                on_token=on_token,
            )

            # no tool calls -> LLM is done, log the final step and return
            if not resp.tool_calls:
                self.messages.append(resp.message)
                self._log_step(self._step_number, len(full_msgs), est_tokens,
                               resp, [], step_start)
                return resp.content

            # tool calls -> execute (async gather for parallelism)
            self.messages.append(resp.message)

            try:
                results = await self._exec_tools_async(resp.tool_calls, on_tool)
                for tc, (result, _elapsed, _success) in results:
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
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
            await asyncio.to_thread(self.context.maybe_compress, self.messages, self.llm)

        return "(reached maximum tool-call rounds)"

    async def _exec_tool(self, tc: Any) -> tuple[str, float, bool]:
        """Execute a single tool call. Returns (result, elapsed_ms, success)."""
        tool = self._tool_by_name.get(tc.name)
        if tool is None:
            logger.warning("Unknown tool requested: %s", tc.name)
            return f"Error: unknown tool '{tc.name}'", 0, False
        # validate arguments first so a TypeError raised *inside* the tool isn't
        # mislabelled as a bad-arguments error from the caller
        try:
            inspect.signature(tool.execute).bind(**tc.arguments)
        except TypeError as e:
            logger.debug("Bad arguments for %s: %s", tc.name, e)
            return f"Error: bad arguments for {tc.name}: {e}", 0, False

        # ---- security review ----
        if self.guard is not None:
            decision = self.guard.review(tc.name, tc.arguments)
            if not decision.allowed:
                return f"[Security] Blocked: {decision.reason}", 0, False

        t0 = time.monotonic()
        try:
            result = await tool.execute(**tc.arguments)
            # ---- output sanitisation ----
            if self.guard is not None:
                result = self.guard.sanitize(result)
            elapsed = (time.monotonic() - t0) * 1000
            success = not result.startswith("Error")
            if not success:
                logger.debug("Tool %s failed: %s", tc.name, result[:200])
            return result, elapsed, success
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            logger.exception("Tool %s raised exception", tc.name)
            return f"Error executing {tc.name}: {e}", elapsed, False

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
            if name in ("read_file", "grep", "glob"):
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
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "[interrupted]",
                })

    def reset(self):
        """Clear conversation history."""
        self.messages.clear()
        self._step_number = 0
        self._memory_prompt = ""
        self._memory_context_loaded = False
        self._memory_finalized = False
        self.session_id = self._new_session_id()

    @staticmethod
    def _new_session_id() -> str:
        return f"session_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    def learn(self) -> list:
        """Extract durable memory from this session once, without blocking shutdown."""
        if self.memory is None or self._memory_finalized:
            return []
        self._memory_finalized = True
        try:
            return self.memory.learn(self.messages, self.session_id)
        except Exception:
            logger.warning("Failed to learn from session", exc_info=True)
            return []

    def _load_memory_context(self, user_input: str) -> None:
        if self.memory is None or self._memory_context_loaded:
            return
        self._memory_context_loaded = True
        try:
            self._memory_prompt = self.memory.build_prompt(user_input)
        except Exception:
            logger.warning("Failed to retrieve cross-session memory", exc_info=True)
            self._memory_prompt = ""

    def close(self):
        """Learn from the conversation and close replay. Safe to call repeatedly."""
        self.learn()
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
        tools = role_tools(role, self.tools)
        # remove agent tool to prevent infinite recursion
        tools = [t for t in tools if t.name != "agent"]

        sub = Agent(
            llm=self.llm,
            tools=tools,
            max_context_tokens=self.context.max_tokens,
            max_rounds=min(self.max_rounds, 15),
            replay=False,  # sub-agents don't write their own replay logs
            guard=self.guard,  # inherit parent's security policy
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

    async def _review(self, executor_result: str, task: str) -> str:
        """Run a lightweight reviewer pass on executor output."""
        review_prompt = (
            f"Task: {task}\n\n"
            f"Executor output:\n{executor_result[:3000]}\n\n"
            f"Review the above. Report PASS or list specific issues."
        )
        tools = role_tools(AgentRole.REVIEWER, self.tools)
        tools = [t for t in tools if t.name != "agent"]

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
        record = StepRecord(
            step=step,
            messages_count=msg_count,
            estimated_input_tokens=est_tokens,
            llm_response=resp,
            tool_executions=execs,
            step_duration_ms=round(step_duration, 2),
        )
        self._replay.log(record)
