"""Opt-in synthetic provider smoke test; no project content or filesystem tools."""

import os

import pytest

from corecoder.config import Config
from corecoder.llm import LLM, LiteLLM


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
