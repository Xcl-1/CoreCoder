"""Tests for the LiteLLM backend."""

import types as builtin_types
from unittest import mock

import pytest

from corecoder.config import Config
from corecoder.llm import LLM, LiteLLM, LLMResponse

# ---------------------------------------------------------------------------
# Fake streaming response (matches OpenAI stream chunk format)
# ---------------------------------------------------------------------------


class _Delta:
    def __init__(self, content=None, tool_calls=None, reasoning_content=None):
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content


class _Choice:
    def __init__(self, delta, finish_reason=None):
        self.delta = delta
        self.finish_reason = finish_reason


class _Usage:
    def __init__(self, prompt=10, completion=5):
        self.prompt_tokens = prompt
        self.completion_tokens = completion


class _Chunk:
    def __init__(self, content=None, usage=None, tool_calls=None, reasoning_content=None, finish_reason=None):
        has_choice = content or tool_calls or reasoning_content or finish_reason
        self.choices = [
            _Choice(
                _Delta(content=content, tool_calls=tool_calls, reasoning_content=reasoning_content),
                finish_reason=finish_reason,
            )
        ] if has_choice else []
        self.usage = usage


def _make_stream(contents, usage=None):
    """Create a fake stream from a list of content strings."""
    chunks = [_Chunk(content=c) for c in contents]
    if usage:
        chunks.append(_Chunk(usage=usage))
    else:
        chunks.append(_Chunk(usage=_Usage()))
    return iter(chunks)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _install_fake_litellm(stream_contents=None):
    import sys

    fake = builtin_types.ModuleType("litellm")
    if stream_contents is None:
        stream_contents = ["hello", " world"]
    fake.completion = mock.MagicMock(
        return_value=_make_stream(stream_contents)
    )
    sys.modules["litellm"] = fake
    return fake


def _uninstall_fake_litellm():
    import sys

    sys.modules.pop("litellm", None)


# ---------------------------------------------------------------------------
# LiteLLM class basics
# ---------------------------------------------------------------------------


class TestLiteLLMClass:
    def test_extends_llm(self):
        assert issubclass(LiteLLM, LLM)

    def test_init_does_not_create_openai_client(self):
        llm = LiteLLM(model="anthropic/claude-3-haiku")
        assert not hasattr(llm, "client") or llm.__dict__.get("client") is None

    def test_init_stores_model(self):
        llm = LiteLLM(model="bedrock/anthropic.claude-v2", api_key="k")
        assert llm.model == "bedrock/anthropic.claude-v2"

    def test_init_stores_api_key(self):
        llm = LiteLLM(model="x", api_key="sk-test")
        assert llm.api_key == "sk-test"

    def test_init_stores_base_url(self):
        llm = LiteLLM(model="x", base_url="http://localhost:4000")
        assert llm.base_url == "http://localhost:4000"

    def test_init_stores_extra_kwargs(self):
        llm = LiteLLM(model="x", temperature=0.7, max_tokens=2048)
        assert llm.extra == {"temperature": 0.7, "max_tokens": 2048}

    def test_token_counters_start_at_zero(self):
        llm = LiteLLM(model="x")
        assert llm.total_prompt_tokens == 0
        assert llm.total_completion_tokens == 0


def test_openai_compatible_client_is_lazy_and_forked_independently():
    llm = LLM(model="test-model", api_key="test-key", base_url="http://localhost:1234", max_tokens=42)

    forked = llm.fork()

    assert llm.client is None
    assert forked.client is None
    assert forked is not llm
    assert forked.model == "test-model"
    assert forked.api_key == "test-key"
    assert forked.base_url == "http://localhost:1234"
    assert forked.extra == {"max_tokens": 42}


# ---------------------------------------------------------------------------
# _call_with_retry
# ---------------------------------------------------------------------------


class TestCallWithRetry:
    def setup_method(self):
        self.fake = _install_fake_litellm()

    def teardown_method(self):
        _uninstall_fake_litellm()

    def test_passes_drop_params(self):
        llm = LiteLLM(model="openai/gpt-4o")
        llm._call_with_retry({"model": "openai/gpt-4o", "messages": [], "stream": True})
        call_kwargs = self.fake.completion.call_args[1]
        assert call_kwargs["drop_params"] is True

    def test_forwards_api_key(self):
        llm = LiteLLM(model="x", api_key="sk-test")
        llm._call_with_retry({"model": "x", "messages": [], "stream": True})
        call_kwargs = self.fake.completion.call_args[1]
        assert call_kwargs["api_key"] == "sk-test"

    def test_omits_api_key_when_none(self):
        llm = LiteLLM(model="x")
        llm._call_with_retry({"model": "x", "messages": [], "stream": True})
        call_kwargs = self.fake.completion.call_args[1]
        assert "api_key" not in call_kwargs

    def test_forwards_api_base(self):
        llm = LiteLLM(model="x", base_url="http://proxy:4000")
        llm._call_with_retry({"model": "x", "messages": [], "stream": True})
        call_kwargs = self.fake.completion.call_args[1]
        assert call_kwargs["api_base"] == "http://proxy:4000"

    def test_omits_api_base_when_none(self):
        llm = LiteLLM(model="x")
        llm._call_with_retry({"model": "x", "messages": [], "stream": True})
        call_kwargs = self.fake.completion.call_args[1]
        assert "api_base" not in call_kwargs


# ---------------------------------------------------------------------------
# chat() end-to-end (mocked)
# ---------------------------------------------------------------------------


class TestChat:
    def setup_method(self):
        self.fake = _install_fake_litellm(["part1", "part2"])

    def teardown_method(self):
        _uninstall_fake_litellm()

    def test_returns_llm_response(self):
        llm = LiteLLM(model="openai/gpt-4o")
        result = llm.chat(messages=[{"role": "user", "content": "hi"}])
        assert isinstance(result, LLMResponse)
        assert result.content == "part1part2"

    def test_tracks_token_usage(self):
        llm = LiteLLM(model="openai/gpt-4o")
        result = llm.chat(messages=[{"role": "user", "content": "hi"}])
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 5
        assert llm.total_prompt_tokens == 10
        assert llm.total_completion_tokens == 5

    def test_on_token_callback(self):
        llm = LiteLLM(model="openai/gpt-4o")
        tokens = []
        llm.chat(
            messages=[{"role": "user", "content": "hi"}],
            on_token=lambda t: tokens.append(t),
        )
        assert tokens == ["part1", "part2"]

    def test_model_forwarded(self):
        llm = LiteLLM(model="anthropic/claude-3-haiku")
        llm.chat(messages=[{"role": "user", "content": "hi"}])
        call_kwargs = self.fake.completion.call_args[1]
        assert call_kwargs["model"] == "anthropic/claude-3-haiku"

    def test_requests_usage_via_stream_options(self):
        """chat() must ask for usage stats, otherwise token tracking stays zero."""
        llm = LiteLLM(model="openai/gpt-4o")
        llm.chat(messages=[{"role": "user", "content": "hi"}])
        call_kwargs = self.fake.completion.call_args[1]
        assert call_kwargs["stream_options"] == {"include_usage": True}

    def test_preserves_reasoning_and_finish_reason(self):
        self.fake.completion.return_value = iter([
            _Chunk(reasoning_content="private reasoning"),
            _Chunk(content="answer", finish_reason="stop"),
            _Chunk(usage=_Usage()),
        ])
        llm = LiteLLM(model="deepseek/deepseek-v4-flash")
        result = llm.chat(messages=[{"role": "user", "content": "hi"}])

        assert result.content == "answer"
        assert result.reasoning_content == "private reasoning"
        assert result.finish_reason == "stop"
        assert result.message["reasoning_content"] == "private reasoning"


def test_openai_compatible_chat_preserves_reasoning_and_finish_reason():
    llm = LLM.__new__(LLM)
    llm.model = "deepseek-v4-flash"
    llm.extra = {}
    llm.total_prompt_tokens = 0
    llm.total_completion_tokens = 0
    llm._call_with_retry = lambda _params: iter([
        _Chunk(reasoning_content="private reasoning"),
        _Chunk(content="answer", finish_reason="stop"),
        _Chunk(usage=_Usage()),
    ])

    result = llm.chat(messages=[{"role": "user", "content": "hi"}])

    assert result.content == "answer"
    assert result.reasoning_content == "private reasoning"
    assert result.finish_reason == "stop"


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


class TestConfigProvider:
    def test_default_provider_is_openai(self):
        config = Config()
        assert config.provider == "openai"

    def test_provider_from_env(self):
        with mock.patch.dict("os.environ", {"CORECODER_PROVIDER": "litellm"}, clear=False):
            config = Config.from_env()
            assert config.provider == "litellm"

    def test_cli_picks_litellm_class(self):
        from corecoder.llm import LiteLLM
        config = Config(provider="litellm", model="anthropic/claude-3-haiku", api_key="k")
        llm_cls = LiteLLM if config.provider == "litellm" else LLM
        assert llm_cls is LiteLLM


# ---------------------------------------------------------------------------
# Multi-provider model strings
# ---------------------------------------------------------------------------


class TestMultiProvider:
    def setup_method(self):
        self.fake = _install_fake_litellm(["ok"])

    def teardown_method(self):
        _uninstall_fake_litellm()

    @pytest.mark.parametrize(
        "model",
        [
            "openai/gpt-4o",
            "anthropic/claude-3-haiku",
            "bedrock/anthropic.claude-v2",
            "vertex_ai/gemini-pro",
            "groq/llama3-70b-8192",
            "ollama/llama3",
            "azure/gpt-4o",
        ],
    )
    def test_model_string_forwarded(self, model):
        llm = LiteLLM(model=model)
        llm.chat(messages=[{"role": "user", "content": "hi"}])
        call_kwargs = self.fake.completion.call_args[1]
        assert call_kwargs["model"] == model
