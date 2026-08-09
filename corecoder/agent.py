"""Core agent loop.

This is the heart of CoreCoder.  The pattern is simple:

    user message -> LLM (with tools) -> tool calls? -> execute -> loop
                                      -> text reply? -> return to user

It keeps looping until the LLM responds with plain text (no tool calls),
which means it's done working and ready to report back.
"""

import asyncio
import inspect
import time
from .llm import LLM
from .models import ToolExecRecord, StepRecord
from .tools import ALL_TOOLS
from .tools.base import Tool
from .tools.agent import AgentTool
from .prompt import system_prompt
from .context import ContextManager, estimate_tokens
from .replay import ReplayLogger


class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: list[Tool] | None = None,
        max_context_tokens: int = 128_000,
        max_rounds: int = 50,
        replay: bool = True,
    ):
        self.llm = llm
        self.tools = tools if tools is not None else ALL_TOOLS
        self._tool_by_name = {t.name: t for t in self.tools}
        self.messages: list[dict] = []
        self.context = ContextManager(max_tokens=max_context_tokens)
        self.max_rounds = max_rounds
        self._system = system_prompt(self.tools)
        self._step_number = 0

        # replay log — on by default in production, off in tests
        self._replay = ReplayLogger() if replay else None
        if self._replay:
            self._replay.open()

        # wire up sub-agent capability
        for t in self.tools:
            if isinstance(t, AgentTool):
                t._parent_agent = self

    def _full_messages(self) -> list[dict]:
        return [{"role": "system", "content": self._system}] + self.messages

    def _tool_schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools]

    async def chat(self, user_input: str, on_token=None, on_tool=None) -> str:
        """Process one user message. May involve multiple LLM/tool rounds."""
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

    async def _exec_tool(self, tc) -> tuple[str, float, bool]:
        """Execute a single tool call. Returns (result, elapsed_ms, success)."""
        tool = self._tool_by_name.get(tc.name)
        if tool is None:
            return f"Error: unknown tool '{tc.name}'", 0, False
        # validate arguments first so a TypeError raised *inside* the tool isn't
        # mislabelled as a bad-arguments error from the caller
        try:
            inspect.signature(tool.execute).bind(**tc.arguments)
        except TypeError as e:
            return f"Error: bad arguments for {tc.name}: {e}", 0, False
        t0 = time.monotonic()
        try:
            result = await tool.execute(**tc.arguments)
            elapsed = (time.monotonic() - t0) * 1000
            success = not result.startswith("Error")
            return result, elapsed, success
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            return f"Error executing {tc.name}: {e}", elapsed, False

    async def _exec_tools_async(self, tool_calls, on_tool=None) -> list:
        """Run all tool calls concurrently via asyncio.gather.

        Replaces the old ThreadPoolExecutor-based parallel execution with
        native async concurrency — single-tool and multi-tool paths are
        unified into one code path.
        """
        for tc in tool_calls:
            if on_tool:
                on_tool(tc.name, tc.arguments)

        async def _run_one(tc):
            return tc, await self._exec_tool(tc)

        return await asyncio.gather(*[_run_one(tc) for tc in tool_calls])

    def _answer_pending_tool_calls(self, tool_calls):
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

    def close(self):
        """Close the replay log file. Safe to call multiple times."""
        if self._replay:
            self._replay.close()

    def _log_step(self, step: int, msg_count: int, est_tokens: int,
                  resp, results, step_start: float):
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
