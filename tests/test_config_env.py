"""Tests for Config.from_env(): env vars, .env discovery, and dir handling.

Uses pytest's tmp_path (temporary directory) and monkeypatch (env vars),
isolated from any real .env / environment on the developer's machine.
"""

from pathlib import Path

import pytest

import corecoder.config as config_module
from corecoder.config import Config

# Every env var Config.from_env() reads.
ENV_KEYS = [
    "CORECODER_API_KEY",
    "OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "CORECODER_MODEL",
    "CORECODER_MAX_TOKENS",
    "CORECODER_TEMPERATURE",
    "CORECODER_MAX_CONTEXT",
    "CORECODER_PROVIDER",
    "CORECODER_MEMORY",
    "CORECODER_MEMORY_TOP_K",
    "CORECODER_MEMORY_DIR",
    "CORECODER_SKILLS",
    "CORECODER_SKILLS_DIR",
    "CORECODER_SKILL_TOP_K",
    "CORECODER_SKILL_MAX_ACTIVE",
    "CORECODER_SKILL_PROMPT_CHARS",
    "OPENAI_BASE_URL",
    "CORECODER_BASE_URL",
]

# The real loader, captured before the autouse fixture stubs it.
_REAL_LOAD_DOTENV = config_module._load_dotenv


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    """Clear all relevant env vars and disable .env loading by default.

    This guarantees tests are hermetic even if the repo or home dir has a
    stray .env (the repo root currently does).
    """
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(config_module, "_load_dotenv", lambda: None)


@pytest.fixture
def with_real_dotenv(monkeypatch):
    """Re-enable real .env discovery (paired with monkeypatch.chdir(tmp_path))."""
    monkeypatch.setattr(config_module, "_load_dotenv", _REAL_LOAD_DOTENV)


def _write_dotenv(directory: Path, content: str) -> Path:
    dotenv = directory / ".env"
    dotenv.write_text(content, encoding="utf-8")
    return dotenv


# --- env vars ----------------------------------------------------------

def test_env_api_key_priority(monkeypatch):
    monkeypatch.setenv("CORECODER_API_KEY", "corercoder-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    assert Config.from_env().api_key == "corercoder-key"


def test_env_api_key_fallback(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    assert Config.from_env().api_key == "openai-key"

    monkeypatch.delenv("OPENAI_API_KEY")
    assert Config.from_env().api_key == "deepseek-key"


def test_env_full_override(monkeypatch):
    monkeypatch.setenv("CORECODER_MODEL", "my-model")
    monkeypatch.setenv("CORECODER_MAX_TOKENS", "2048")
    monkeypatch.setenv("CORECODER_TEMPERATURE", "1.5")
    monkeypatch.setenv("CORECODER_MAX_CONTEXT", "32768")
    monkeypatch.setenv("CORECODER_PROVIDER", "litellm")
    monkeypatch.setenv("CORECODER_MEMORY_TOP_K", "8")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example.com")

    c = Config.from_env()
    assert c.model == "my-model"
    assert c.max_tokens == 2048
    assert c.temperature == 1.5
    assert c.max_context_tokens == 32768
    assert c.provider == "litellm"
    assert c.memory_top_k == 8
    assert c.base_url == "https://proxy.example.com"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1", True),
        ("true", True),
        ("yes", True),
        ("0", False),
        ("false", False),
        ("no", False),
    ],
)
def test_memory_enabled_parsing(monkeypatch, raw, expected):
    monkeypatch.setenv("CORECODER_MEMORY", raw)
    assert Config.from_env().memory_enabled is expected


@pytest.mark.parametrize("raw,expected", [("1", True), ("true", True), ("0", False), ("no", False)])
def test_skills_enabled_parsing(monkeypatch, raw, expected):
    monkeypatch.setenv("CORECODER_SKILLS", raw)
    assert Config.from_env().skills_enabled is expected


def test_skill_configuration(monkeypatch, tmp_path):
    monkeypatch.setenv("CORECODER_SKILLS_DIR", str(tmp_path / "skills"))
    monkeypatch.setenv("CORECODER_SKILL_TOP_K", "7")
    monkeypatch.setenv("CORECODER_SKILL_MAX_ACTIVE", "3")
    monkeypatch.setenv("CORECODER_SKILL_PROMPT_CHARS", "9000")
    config = Config.from_env()
    assert config.skills_dir == tmp_path / "skills"
    assert config.skill_top_k == 7
    assert config.skill_max_active == 3
    assert config.skill_prompt_chars == 9000


# --- .env file in temporary directory ----------------------------------

def test_dotenv_in_cwd(tmp_path, monkeypatch, with_real_dotenv):
    _write_dotenv(
        tmp_path,
        "CORECODER_MODEL=dotenv-model\n"
        "OPENAI_API_KEY=dotenv-key\n"
        "CORECODER_MAX_TOKENS=2048\n",
    )
    monkeypatch.chdir(tmp_path)

    c = Config.from_env()
    assert c.model == "dotenv-model"
    assert c.api_key == "dotenv-key"
    assert c.max_tokens == 2048


def test_dotenv_found_in_parent_dir(tmp_path, monkeypatch, with_real_dotenv):
    """cwd has no .env, but a parent dir does — _load_dotenv walks up."""
    _write_dotenv(tmp_path, "CORECODER_MODEL=parent-model\n")
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)

    assert Config.from_env().model == "parent-model"


def test_dotenv_nearest_wins(tmp_path, monkeypatch, with_real_dotenv):
    """Ancestor .env is ignored when a closer one exists."""
    _write_dotenv(tmp_path, "CORECODER_MODEL=far-model\n")
    sub = tmp_path / "sub"
    sub.mkdir()
    _write_dotenv(sub, "CORECODER_MODEL=near-model\n")
    monkeypatch.chdir(sub)

    assert Config.from_env().model == "near-model"


def test_env_var_overrides_dotenv(tmp_path, monkeypatch, with_real_dotenv):
    """load_dotenv(override=False): a real env var beats the .env value."""
    _write_dotenv(tmp_path, "CORECODER_MODEL=dotenv-model\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CORECODER_MODEL", "env-model")

    assert Config.from_env().model == "env-model"


def test_no_dotenv_uses_defaults(tmp_path, monkeypatch, with_real_dotenv):
    """Empty temp dir, no .env anywhere up the chain -> defaults."""
    deep = tmp_path / "x" / "y" / "z"
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)

    c = Config.from_env()
    assert c.model == "gpt-5.5"
    assert c.max_tokens == 4096
    assert c.provider == "openai"


# --- memory_dir --------------------------------------------------------

def test_memory_dir_from_env(tmp_path, monkeypatch):
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    monkeypatch.setenv("CORECODER_MEMORY_DIR", str(mem_dir))

    c = Config.from_env()
    assert c.memory_dir == mem_dir
    assert c.memory_dir.is_dir()


def test_memory_dir_default_expands_home(monkeypatch):
    fake_home = Path.cwd() / "fakehome"
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))

    c = Config.from_env()
    assert c.memory_dir == fake_home / ".corecoder" / "memory"
    assert c.memory_dir.is_absolute()


# --- validation --------------------------------------------------------

@pytest.mark.parametrize(
    "key,raw",
    [
        ("CORECODER_MAX_TOKENS", "abc"),
        ("CORECODER_MAX_TOKENS", "0"),
        ("CORECODER_TEMPERATURE", "hot"),
        ("CORECODER_TEMPERATURE", "3.5"),
        ("CORECODER_MAX_CONTEXT", "100"),
        ("CORECODER_PROVIDER", "anthropic"),
        ("CORECODER_MEMORY", "maybe"),
        ("CORECODER_MEMORY_TOP_K", "99"),
        ("CORECODER_SKILLS", "maybe"),
        ("CORECODER_SKILL_TOP_K", "0"),
        ("CORECODER_SKILL_MAX_ACTIVE", "8"),
        ("CORECODER_SKILL_PROMPT_CHARS", "100"),
    ],
)
def test_invalid_env_raises(monkeypatch, key, raw):
    monkeypatch.setenv(key, raw)
    with pytest.raises(ValueError):
        Config.from_env()
