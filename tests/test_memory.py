"""Tests for cross-session memory storage, extraction and integration."""

from __future__ import annotations

import json

import corecoder.cli as cli_module
from corecoder.agent import Agent
from corecoder.config import Config
from corecoder.llm import LLM
from corecoder.memory import Memory, MemoryEngine, MemoryRetriever, MemoryStore
from corecoder.memory.extractor import MemoryExtractor
from corecoder.memory.index import MemoryIndex
from corecoder.models import LLMResponse


def _memory(memory_id="prefer-pytest", **updates):
    data = {
        "id": memory_id,
        "title": "Prefer pytest fixtures",
        "description": "Use pytest fixtures for tests",
        "content": "Use tmp_path and monkeypatch instead of unittest setup.",
        "type": "user",
        "scope": "global",
        "keywords": ["pytest", "fixtures", "测试"],
    }
    data.update(updates)
    return Memory(**data)


class _FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    def chat(self, **_kwargs):
        return LLMResponse(content=self.responses.pop(0))


def test_store_roundtrip_and_index(tmp_path):
    store = MemoryStore(tmp_path)
    stored = store.save(_memory())

    assert stored.id == "prefer-pytest"
    assert store.get("prefer-pytest") == stored
    assert len(store.list()) == 1

    path = MemoryIndex(tmp_path).rebuild(store.list())
    index = path.read_text(encoding="utf-8")
    assert "[Prefer pytest fixtures](prefer-pytest.md)" in index


def test_store_skips_corrupt_markdown(tmp_path):
    (tmp_path / "broken.md").write_text("not front matter", encoding="utf-8")
    assert MemoryStore(tmp_path).list() == []


def test_store_and_engine_honor_memory_dir_env(tmp_path, monkeypatch):
    configured = tmp_path / "configured-memory"
    monkeypatch.setenv("CORECODER_MEMORY_DIR", str(configured))

    store = MemoryStore()
    engine = MemoryEngine(_FakeLLM([]), project_path=tmp_path)

    assert store.root == configured.resolve()
    assert engine.store.root == configured.resolve()


def test_retriever_filters_project_scope_and_matches_chinese(tmp_path):
    project = tmp_path / "project-a"
    other = tmp_path / "project-b"
    memories = [
        _memory(),
        _memory(
            "project-layout",
            title="项目测试约定",
            description="测试放在 tests 目录",
            content="所有测试使用 pytest。",
            type="project",
            scope="project",
            project_path=str(project.resolve()),
            keywords=["测试", "pytest"],
        ),
    ]

    current = MemoryRetriever().retrieve("请生成 pytest 测试", memories, project_path=project)
    elsewhere = MemoryRetriever().retrieve("请生成 pytest 测试", memories, project_path=other)

    assert {item.memory.id for item in current} == {"prefer-pytest", "project-layout"}
    assert {item.memory.id for item in elsewhere} == {"prefer-pytest"}


def test_extractor_validates_json_and_drops_secrets():
    payload = [
        {
            "action": "create",
            "title": "Prefer concise answers",
            "description": "Keep answers short",
            "content": "Use concise explanations.",
            "type": "user",
            "scope": "global",
            "keywords": ["concise"],
            "confidence": 0.9,
            "evidence": "Please remember: I prefer concise answers.",
        },
        {
            "action": "create",
            "title": "Credential",
            "description": "API key: sk-abcdefghijklmnop",
            "content": "Remember it",
            "type": "reference",
            "scope": "global",
            "keywords": [],
            "confidence": 0.9,
            "evidence": "Please remember: API key: sk-abcdefghijklmnop",
        },
    ]
    extractor = MemoryExtractor(_FakeLLM([f"```json\n{json.dumps(payload)}\n```"]))

    result = extractor.extract(
        [
            {
                "role": "user",
                "content": ("Please remember: I prefer concise answers.\nPlease remember: API key: sk-abcdefghijklmnop"),
            },
            {"role": "assistant", "content": "Okay"},
        ],
        [],
    )

    assert [item.title for item in result] == ["Prefer concise answers"]


def test_extractor_retries_and_accepts_wrapped_object():
    payload = {
        "memories": [
            {
                "action": "create",
                "title": "Prefer concise answers",
                "description": "Keep answers short",
                "content": "Use concise explanations.",
                "type": "user",
                "scope": "global",
                "keywords": ["concise"],
                "confidence": 0.9,
                "evidence": "Please remember: I prefer concise answers.",
            }
        ]
    }
    llm = _FakeLLM(["This is not JSON", f"Result:\n```json\n{json.dumps(payload)}\n```"])
    extractor = MemoryExtractor(llm)

    result = extractor.extract(
        [
            {"role": "user", "content": "Please remember: I prefer concise answers."},
            {"role": "assistant", "content": "I will consider it at session end."},
        ],
        [],
    )

    assert [item.title for item in result] == ["Prefer concise answers"]
    assert llm.responses == []


def test_extractor_rejects_one_off_task_instruction():
    payload = [
        {
            "action": "create",
            "title": "Only show code",
            "description": "Do not modify project files",
            "content": "Only show code and do not edit files.",
            "type": "user",
            "scope": "global",
            "keywords": ["code only", "read-only"],
            "confidence": 0.9,
            "evidence": "请只展示代码，不修改项目文件。",
        }
    ]
    extractor = MemoryExtractor(_FakeLLM([json.dumps(payload)]))

    result = extractor.extract(
        [
            {
                "role": "user",
                "content": "请只展示代码，不修改项目文件。为一个函数生成测试。",
            },
            {"role": "assistant", "content": "Here is the code."},
        ],
        [],
    )

    assert result == []


def test_extractor_rejects_reusable_summary_request_as_durable_fact():
    request = "请运行 pytest，并总结以后验证记忆系统时可以重复使用的步骤。"
    payload = [
        {
            "action": "create",
            "title": f"False {memory_type} memory",
            "description": "The user wants reusable verification steps",
            "content": "Summarize the verification workflow for future sessions.",
            "type": memory_type,
            "scope": "project",
            "keywords": ["pytest", "verification"],
            "confidence": 0.9,
            "evidence": request,
        }
        for memory_type in ("user", "profile", "feedback", "project", "reference")
    ]
    extractor = MemoryExtractor(_FakeLLM([json.dumps(payload, ensure_ascii=False)]))

    result = extractor.extract(
        [
            {"role": "user", "content": request},
            {"role": "assistant", "content": "测试完成，以下是验证步骤。"},
        ],
        [],
    )

    assert result == []


def test_extractor_keeps_explicit_preference_profile_feedback_and_project_convention():
    evidence = {
        "user": "以后回答技术问题时保持简洁，并优先给出命令。",
        "project": "以后这个项目的测试统一使用 pytest。",
        "profile": "我是后端工程师，主要使用 Python。",
        "feedback": "你刚才的回答太啰嗦了，请直接给出命令。",
    }
    payload = [
        {
            "action": "create",
            "title": f"Valid {memory_type} memory",
            "description": f"Durable {memory_type} information",
            "content": text,
            "type": memory_type,
            "scope": "project" if memory_type == "project" else "global",
            "keywords": [memory_type],
            "confidence": 0.9,
            "evidence": text,
        }
        for memory_type, text in evidence.items()
    ]
    extractor = MemoryExtractor(_FakeLLM([json.dumps(payload, ensure_ascii=False)]))
    messages = [{"role": "user", "content": text} for text in evidence.values()]
    messages.append({"role": "assistant", "content": "收到。"})

    result = extractor.extract(messages, [])

    assert [item.type for item in result] == ["user", "project", "profile", "feedback"]


def test_engine_repairs_and_merges_chinese_preference_update(tmp_path):
    existing = _memory(
        title="Python 测试偏好",
        description="优先使用 pytest fixture",
        content="使用 pytest fixture，不使用 unittest.TestCase。",
        keywords=["pytest", "fixture", "unittest"],
    )
    repaired = {
        "memories": [
            {
                "action": "merge",
                "target_id": "prefer-pytest",
                "title": "Python 测试偏好",
                "description": "使用 fixture，并用 parametrize 覆盖边界情况",
                "content": "使用 pytest fixture；边界情况优先使用 pytest.mark.parametrize。",
                "type": "user",
                "scope": "global",
                "keywords": ["pytest", "fixture", "parametrize"],
                "confidence": 0.95,
                "evidence": "更新我的测试偏好：边界情况优先使用 pytest.mark.parametrize。",
            }
        ]
    }
    llm = _FakeLLM(["无法生成有效 JSON", json.dumps(repaired, ensure_ascii=False)])
    engine = MemoryEngine(llm, root=tmp_path, project_path=tmp_path)
    engine.store.save(existing)
    messages = [
        {
            "role": "user",
            "content": "更新我的测试偏好：边界情况优先使用 pytest.mark.parametrize。",
        },
        {"role": "assistant", "content": "将在会话结束时尝试保存。"},
    ]

    saved = engine.learn(messages, "session-update")

    assert len(saved) == 1
    assert len(engine.store.list()) == 1
    assert "pytest.mark.parametrize" in engine.store.get("prefer-pytest").content


def test_engine_creates_then_merges_memory(tmp_path):
    create = [
        {
            "action": "create",
            "title": "Prefer pytest",
            "description": "Use pytest for tests",
            "content": "Use pytest fixtures.",
            "type": "user",
            "scope": "global",
            "keywords": ["pytest"],
            "confidence": 0.9,
            "evidence": "Please remember: I prefer pytest fixtures.",
        }
    ]
    merge = [
        {
            "action": "merge",
            "target_id": "prefer-pytest",
            "title": "Prefer pytest",
            "description": "Use pytest fixtures and parametrization",
            "content": "Use pytest fixtures and parametrize edge cases.",
            "type": "user",
            "scope": "global",
            "keywords": ["pytest", "parametrize"],
            "confidence": 0.95,
            "evidence": "Update my testing preference: use pytest parametrize for edge cases.",
        }
    ]
    llm = _FakeLLM([json.dumps(create), json.dumps(merge)])
    engine = MemoryEngine(llm, root=tmp_path, project_path=tmp_path)
    first_messages = [
        {"role": "user", "content": "Please remember: I prefer pytest fixtures."},
        {"role": "assistant", "content": "I will consider it at session end."},
    ]
    second_messages = [
        {
            "role": "user",
            "content": "Update my testing preference: use pytest parametrize for edge cases.",
        },
        {"role": "assistant", "content": "I will consider it at session end."},
    ]

    first = engine.learn(first_messages, "session-1")
    second = engine.learn(second_messages, "session-2")

    assert len(first) == len(second) == 1
    stored = engine.store.get("prefer-pytest")
    assert stored is not None
    assert stored.source_sessions == ["session-1", "session-2"]
    assert "parametrize" in stored.content
    assert len(engine.store.list()) == 1
    assert (tmp_path / "MEMORY.md").exists()


def test_engine_builds_bounded_prompt(tmp_path):
    engine = MemoryEngine(_FakeLLM([]), root=tmp_path, project_path=tmp_path, max_prompt_chars=1_000)
    engine.store.save(_memory())

    prompt = engine.build_prompt("Please write pytest tests with fixtures")

    assert "Relevant cross-session memory" in prompt
    assert "Use tmp_path" in prompt
    assert "do not use tools to persist it yourself" in prompt
    assert len(prompt) <= 1_000


def test_engine_injects_management_policy_without_matches(tmp_path):
    engine = MemoryEngine(_FakeLLM([]), root=tmp_path, project_path=tmp_path)

    prompt = engine.build_prompt("There are no memories yet")

    assert "managed automatically by MemoryEngine" in prompt
    assert "Do not inspect, edit, script" in prompt
    assert "do not use tools to persist it yourself" in prompt
    assert "do not claim it is already saved" in prompt
    assert "Relevant cross-session memory" not in prompt


def test_agent_memory_context_is_transient_and_explicit_learning_is_idempotent():
    class _StubMemory:
        def __init__(self):
            self.learn_calls = 0

        def build_prompt(self, query):
            assert query == "write tests"
            return "remember pytest"

        def learn(self, messages, source_session):
            self.learn_calls += 1
            assert messages
            assert source_session
            return []

    memory = _StubMemory()
    agent = Agent(llm=LLM.__new__(LLM), tools=[], replay=False, memory=memory)
    agent._load_memory_context("write tests")
    agent.messages = [{"role": "user", "content": "write tests"}, {"role": "assistant", "content": "done"}]

    assert "remember pytest" in agent._full_messages()[0]["content"]
    assert all("remember pytest" not in str(message) for message in agent.messages)
    agent.learn()
    agent.learn()
    assert memory.learn_calls == 1

    agent.close()
    agent.close()
    assert memory.learn_calls == 1


def test_agent_checkpoint_schedules_each_complete_turn_once(tmp_path):
    class _Worker:
        def __init__(self):
            self.submitted = []

        def submit(self, session_id):
            self.submitted.append(session_id)

        def close(self, **_kwargs):
            pass

    memory = MemoryEngine(_FakeLLM([]), root=tmp_path, project_path=tmp_path)
    worker = _Worker()
    agent = Agent(
        llm=LLM.__new__(LLM),
        tools=[],
        replay=False,
        memory=memory,
        memory_worker=worker,
        session_id="incremental-session",
    )
    agent.messages = [
        {"role": "user", "content": "Remember that I prefer concise replies."},
        {"role": "assistant", "content": "Understood."},
    ]

    agent.checkpoint_memory()
    agent.checkpoint_memory()

    assert worker.submitted == ["incremental-session"]
    assert memory.pending_count() == 1


def test_agent_close_never_learns_or_waits():
    class _Memory:
        def __init__(self):
            self.learn_calls = 0

        def learn(self, *_args, **_kwargs):
            self.learn_calls += 1

    class _Worker:
        def __init__(self):
            self.close_calls = []

        def close(self, **kwargs):
            self.close_calls.append(kwargs)

    memory = _Memory()
    worker = _Worker()
    agent = Agent(
        llm=LLM.__new__(LLM),
        tools=[],
        replay=False,
        memory=memory,
        memory_worker=worker,
    )

    agent.close()

    assert memory.learn_calls == 0
    assert worker.close_calls == [{"wait": False}]


def test_memory_config_from_env(tmp_path, monkeypatch):
    monkeypatch.setattr("corecoder.config._load_dotenv", lambda: None)
    monkeypatch.setenv("CORECODER_MEMORY", "no")
    monkeypatch.setenv("CORECODER_MEMORY_DIR", str(tmp_path))
    monkeypatch.setenv("CORECODER_MEMORY_TOP_K", "7")

    config = Config.from_env()

    assert config.memory_enabled is False
    assert config.memory_dir == tmp_path
    assert config.memory_top_k == 7


def test_disabled_memory_is_not_constructed_and_status_shows_path(tmp_path, monkeypatch):
    config = Config(memory_enabled=False, memory_dir=tmp_path)
    output = []

    class _Console:
        def print(self, message):
            output.append(str(message))

    monkeypatch.setattr(cli_module, "console", _Console())
    agent = Agent(llm=LLM.__new__(LLM), tools=[], replay=False)

    assert cli_module._create_memory_engine(config, object()) is None
    cli_module._show_memory(agent, config)
    assert "disabled" in "\n".join(output)
    assert str(tmp_path) in "\n".join(output)

    output.clear()
    config.memory_enabled = True
    agent.memory = MemoryEngine(_FakeLLM([]), root=tmp_path / "active", project_path=tmp_path)
    cli_module._show_memory(agent, config)
    assert "enabled" in "\n".join(output)
    assert str((tmp_path / "active").resolve()) in "\n".join(output)
