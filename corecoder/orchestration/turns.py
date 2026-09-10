"""LangGraph wrapper for every main-agent and child-agent conversation turn."""

from __future__ import annotations

import enum
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field

TurnTokenCallback = Callable[[str], None] | None
TurnToolCallback = Callable[[str, dict[str, Any]], None] | None
TurnExecutor = Callable[
    [str, TurnTokenCallback, TurnToolCallback, Any],
    Awaitable["TurnExecution"],
]


class TurnStatus(str, enum.Enum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class TurnStage(str, enum.Enum):
    PLAN = "plan"
    EXECUTE = "execute"
    VERIFY = "verify"
    REVIEW = "review"
    FINISHED = "finished"


class TurnExecution(BaseModel):
    """Facts emitted by the guarded raw agent loop."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    answer: str
    status: TurnStatus
    policy_violations: int = Field(default=0, ge=0)


class TurnWorkflowResult(BaseModel):
    """Result of routing one conversation turn through LangGraph."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    turn_id: str
    execution: TurnExecution
    verified: bool
    reviewed: bool
    trace: tuple[TurnStage, ...]


@dataclass(frozen=True)
class _TurnCallbacks:
    on_token: TurnTokenCallback
    on_tool: TurnToolCallback
    routing_context: Any


class _TurnState(TypedDict, total=False):
    turn_id: str
    user_input: str
    execution: TurnExecution
    verified: bool
    reviewed: bool
    trace: list[TurnStage]


class LangGraphTurnOrchestrator:
    """Put every agent loop behind the same explicit graph lifecycle.

    Streaming callbacks and routing objects stay in process-local memory and
    never enter graph state or checkpoints.
    """

    def __init__(self, execute_turn: TurnExecutor):
        self._execute_turn = execute_turn
        self._callbacks: dict[str, _TurnCallbacks] = {}
        self._graph = self._build_graph()

    def _build_graph(self):
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:
            raise RuntimeError(
                "CoreCoder installation is missing its LangGraph dependencies"
            ) from exc

        builder = StateGraph(_TurnState)
        builder.add_node("plan", self._plan)
        builder.add_node("execute", self._execute)
        builder.add_node("verify", self._verify)
        builder.add_node("review", self._review)
        builder.add_edge(START, "plan")
        builder.add_edge("plan", "execute")
        builder.add_edge("execute", "verify")
        builder.add_edge("verify", "review")
        builder.add_edge("review", END)
        return builder.compile()

    @staticmethod
    async def _plan(_state: _TurnState) -> dict:
        return {"trace": [TurnStage.PLAN]}

    async def _execute(self, state: _TurnState) -> dict:
        callbacks = self._callbacks.get(state["turn_id"])
        if callbacks is None:
            raise RuntimeError("turn callback context is unavailable")
        execution = await self._execute_turn(
            state["user_input"],
            callbacks.on_token,
            callbacks.on_tool,
            callbacks.routing_context,
        )
        return {"execution": execution, "trace": [*state["trace"], TurnStage.EXECUTE]}

    @staticmethod
    async def _verify(state: _TurnState) -> dict:
        execution = state["execution"]
        verified = (
            execution.status == TurnStatus.COMPLETED
            and execution.policy_violations == 0
        )
        return {"verified": verified, "trace": [*state["trace"], TurnStage.VERIFY]}

    @staticmethod
    async def _review(state: _TurnState) -> dict:
        return {
            "reviewed": state["verified"],
            "trace": [*state["trace"], TurnStage.REVIEW],
        }

    async def run(
        self,
        user_input: str,
        *,
        on_token: TurnTokenCallback = None,
        on_tool: TurnToolCallback = None,
        routing_context: Any = None,
    ) -> TurnWorkflowResult:
        turn_id = f"turn_{uuid.uuid4().hex[:12]}"
        self._callbacks[turn_id] = _TurnCallbacks(on_token, on_tool, routing_context)
        try:
            state = await self._graph.ainvoke({
                "turn_id": turn_id,
                "user_input": user_input,
                "trace": [],
            })
        finally:
            self._callbacks.pop(turn_id, None)
        return TurnWorkflowResult(
            turn_id=turn_id,
            execution=state["execution"],
            verified=state["verified"],
            reviewed=state["reviewed"],
            trace=(*state["trace"], TurnStage.FINISHED),
        )
