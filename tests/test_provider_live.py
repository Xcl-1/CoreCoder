"""Opt-in synthetic provider smoke test; no project content or filesystem tools."""

import os

import pytest

from corecoder.agent import Agent
from corecoder.config import Config
from corecoder.context_artifacts import ContextArtifactStore
from corecoder.llm import LLM, LiteLLM
from corecoder.tools.base import Tool


class _SyntheticLargeTool(Tool):
    name = "synthetic_large_result"
    description = "Return a large synthetic result containing one hidden marker."
    side_effect = "none"

    def __init__(self):
        self.parameters = {"type": "object", "properties": {}, "required": []}

    def _execute_sync(self) -> str:
        lines = []
        for number in range(400):
            value = "SYNTHETIC_MARKER=cache-roundtrip-2718" if number == 200 else "synthetic-detail"
            lines.append(f"synthetic line {number:03d}: {value} " + "x" * 40)
        return "\n".join(lines)


@pytest.mark.skipif(os.getenv("CORECODER_LIVE_TESTS") != "1", reason="opt-in live provider test")
def test_deepseek_tool_rounds_with_synthetic_context_acknowledgement():
    config = Config.from_env()
    if "deepseek" not in config.model.casefold() or not config.api_key:
        pytest.skip("requires configured DeepSeek model and API key")
    provider = LiteLLM if config.provider == "litellm" else LLM
    llm = provider(model=config.model, api_key=config.api_key, base_url=config.base_url, max_tokens=4096)
    schema = [{"type": "function", "function": {
        "name": "verification_probe", "description": "Read a synthetic verification fact.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }}]
    messages = [
        {"role": "user", "content": "[Conversation summary] This is a synthetic tool protocol test."},
        {"role": "assistant", "content": "Understood. I have the full context."},
        {"role": "user", "content": "Call verification_probe once now. Do not answer without calling it."},
    ]
    try:
        for stage in range(2):
            response = llm.chat(messages=messages, tools=schema)
            assert response.tool_calls, "provider did not execute the requested probe"
            messages.append(response.message)
            for call in response.tool_calls:
                assert call.name == "verification_probe"
                messages.append({"role": "tool", "tool_call_id": call.id, "content": f"probe {stage + 1}: PASS"})
            if stage == 0:
                messages.append({"role": "user", "content": "Call verification_probe once again for the second check."})
        messages.append({"role": "user", "content": "Both probes are complete. Reply only PASS, with no tool calls."})
        response = llm.chat(messages=messages, tools=schema)
        assert not response.tool_calls and response.content.strip() == "PASS"
    finally:
        llm.close()


@pytest.mark.skipif(os.getenv("CORECODER_LIVE_TESTS") != "1", reason="opt-in live provider test")
def test_live_provider_cache_usage_is_recorded():
    config = Config.from_env()
    if "deepseek" not in config.model.casefold() or not config.api_key:
        pytest.skip("requires configured DeepSeek model and API key")
    provider = LiteLLM if config.provider == "litellm" else LLM
    llm = provider(model=config.model, api_key=config.api_key, base_url=config.base_url, max_tokens=16)
    stable_prefix = "\n".join(
        f"synthetic cache line {number:04d}: " + "x" * 64
        for number in range(400)
    )
    messages = [
        {"role": "system", "content": "This is synthetic cache telemetry verification data."},
        {"role": "user", "content": f"Reply only OK after reading this synthetic data:\n{stable_prefix}"},
    ]
    try:
        first = llm.chat(messages=messages)
        second = llm.chat(messages=messages)

        assert first.cache_usage_available is True
        assert second.cache_usage_available is True
        assert second.cached_prompt_tokens > 0
        assert llm.cache_usage_requests == 2
        assert llm.total_cached_prompt_tokens == first.cached_prompt_tokens + second.cached_prompt_tokens
        assert llm.total_cache_miss_prompt_tokens == (
            first.cache_miss_prompt_tokens + second.cache_miss_prompt_tokens
        )
        assert llm.prompt_cache_hit_rate is not None
    finally:
        llm.close()


@pytest.mark.asyncio
@pytest.mark.skipif(os.getenv("CORECODER_LIVE_TESTS") != "1", reason="opt-in live provider test")
async def test_live_agent_externalizes_and_retrieves_hidden_synthetic_evidence(tmp_path):
    config = Config.from_env()
    if "deepseek" not in config.model.casefold() or not config.api_key:
        pytest.skip("requires configured DeepSeek model and API key")
    provider = LiteLLM if config.provider == "litellm" else LLM
    llm = provider(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        max_tokens=1024,
    )
    store = ContextArtifactStore("live-synthetic", root=tmp_path, threshold_chars=1_000)
    agent = Agent(
        llm=llm,
        tools=[_SyntheticLargeTool()],
        artifact_store=store,
        max_rounds=5,
        replay=False,
    )
    try:
        answer = await agent.chat(
            "Call synthetic_large_result exactly once. Its marker value is hidden in the middle "
            "of the result and is not present in this request. After the result is externalized, "
            "call retrieve_context with query SYNTHETIC_MARKER, then report the recovered value."
        )
        tool_messages = [message for message in agent.messages if message.get("role") == "tool"]
        assert any(message["content"].startswith("[Tool Artifact]") for message in tool_messages)
        assert any(message["content"].startswith("[Context Artifact") for message in tool_messages)
        assert "cache-roundtrip-2718" in answer
        assert store.stats()["externalized"] == 1
        assert store.stats()["retrievals"] >= 1
    finally:
        agent.close()
        llm.close()
