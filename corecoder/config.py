"""Configuration - env vars and defaults."""

import os
from dataclasses import dataclass
from pathlib import Path


def resolve_memory_dir(value: str | Path | None = None) -> Path:
    """Resolve the configured memory directory for CLI and library callers."""
    raw = value if value is not None else (os.getenv("CORECODER_MEMORY_DIR") or "~/.corecoder/memory")
    return Path(raw).expanduser().resolve()


def resolve_skill_dir(value: str | Path | None = None) -> Path:
    """Resolve the user-level skill directory without creating it."""
    raw = value if value is not None else (os.getenv("CORECODER_SKILLS_DIR") or "~/.corecoder/skills")
    return Path(raw).expanduser().resolve()


def _load_dotenv():
    """Load .env from cwd, walking up to home dir. No-op if python-dotenv missing."""
    try:
        from dotenv import load_dotenv
        # search cwd first, then parent dirs up to ~
        env_path = Path(".env")
        if not env_path.exists():
            cur = Path.cwd()
            home = Path.home()
            while cur != home and cur != cur.parent:
                candidate = cur / ".env"
                if candidate.exists():
                    env_path = candidate
                    break
                cur = cur.parent
        load_dotenv(env_path, override=False)
    except ImportError:
        pass  # python-dotenv not installed, silently skip


@dataclass
class Config:
    model: str = "gpt-5.5"
    api_key: str = ""
    base_url: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.0
    max_context_tokens: int = 128_000
    provider: str = "openai"
    memory_enabled: bool = True
    memory_dir: Path = Path.home() / ".corecoder" / "memory"
    memory_top_k: int = 5
    skills_enabled: bool = True
    skills_dir: Path = Path.home() / ".corecoder" / "skills"
    skill_top_k: int = 10
    skill_max_active: int = 2
    skill_prompt_chars: int = 6_000

    @classmethod
    def from_env(cls) -> "Config":
        # load .env if present (won't override existing env vars)
        _load_dotenv()
        # pick up common env vars automatically
        api_key = (
            os.getenv("CORECODER_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
            or ""
        )
        max_tokens_raw = os.getenv("CORECODER_MAX_TOKENS", "4096")
        temperature_raw = os.getenv("CORECODER_TEMPERATURE", "0")
        max_context_raw = os.getenv("CORECODER_MAX_CONTEXT", "128000")
        provider = os.getenv("CORECODER_PROVIDER", "openai")
        memory_raw = os.getenv("CORECODER_MEMORY", "1").strip().lower()
        memory_top_k_raw = os.getenv("CORECODER_MEMORY_TOP_K", "5")
        skills_raw = os.getenv("CORECODER_SKILLS", "1").strip().lower()
        skill_top_k_raw = os.getenv("CORECODER_SKILL_TOP_K", "10")
        skill_max_active_raw = os.getenv("CORECODER_SKILL_MAX_ACTIVE", "2")
        skill_prompt_chars_raw = os.getenv("CORECODER_SKILL_PROMPT_CHARS", "6000")

        # --- validation --------------------------------------------------
        try:
            max_tokens = int(max_tokens_raw)
        except ValueError:
            raise ValueError(
                f"CORECODER_MAX_TOKENS must be an integer, got: {max_tokens_raw!r}"
            )
        if max_tokens < 1:
            raise ValueError(
                f"CORECODER_MAX_TOKENS must be positive, got: {max_tokens}"
            )

        try:
            temperature = float(temperature_raw)
        except ValueError:
            raise ValueError(
                f"CORECODER_TEMPERATURE must be a number, got: {temperature_raw!r}"
            )
        if not (0.0 <= temperature <= 2.0):
            raise ValueError(
                f"CORECODER_TEMPERATURE must be 0.0–2.0, got: {temperature}"
            )

        try:
            max_context_tokens = int(max_context_raw)
        except ValueError:
            raise ValueError(
                f"CORECODER_MAX_CONTEXT must be an integer, got: {max_context_raw!r}"
            )
        if max_context_tokens < 1024:
            raise ValueError(
                f"CORECODER_MAX_CONTEXT must be at least 1024, got: {max_context_tokens}"
            )

        if provider not in ("openai", "litellm"):
            raise ValueError(
                f"CORECODER_PROVIDER must be 'openai' or 'litellm', got: {provider!r}"
            )

        if memory_raw not in ("1", "true", "yes", "0", "false", "no"):
            raise ValueError(
                f"CORECODER_MEMORY must be a boolean, got: {memory_raw!r}"
            )
        try:
            memory_top_k = int(memory_top_k_raw)
        except ValueError:
            raise ValueError(
                f"CORECODER_MEMORY_TOP_K must be an integer, got: {memory_top_k_raw!r}"
            )
        if not (1 <= memory_top_k <= 20):
            raise ValueError(
                f"CORECODER_MEMORY_TOP_K must be 1-20, got: {memory_top_k}"
            )
        if skills_raw not in ("1", "true", "yes", "0", "false", "no"):
            raise ValueError(
                f"CORECODER_SKILLS must be a boolean, got: {skills_raw!r}"
            )
        try:
            skill_top_k = int(skill_top_k_raw)
            skill_max_active = int(skill_max_active_raw)
            skill_prompt_chars = int(skill_prompt_chars_raw)
        except ValueError as exc:
            raise ValueError("Skill limits must be integers") from exc
        if not (1 <= skill_top_k <= 50):
            raise ValueError(f"CORECODER_SKILL_TOP_K must be 1-50, got: {skill_top_k}")
        if not (1 <= skill_max_active <= 5):
            raise ValueError(
                f"CORECODER_SKILL_MAX_ACTIVE must be 1-5, got: {skill_max_active}"
            )
        if not (1000 <= skill_prompt_chars <= 50000):
            raise ValueError(
                "CORECODER_SKILL_PROMPT_CHARS must be 1000-50000, "
                f"got: {skill_prompt_chars}"
            )

        return cls(
            model=os.getenv("CORECODER_MODEL", "gpt-5.5"),
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("CORECODER_BASE_URL"),
            max_tokens=max_tokens,
            temperature=temperature,
            max_context_tokens=max_context_tokens,
            provider=provider,
            memory_enabled=memory_raw in ("1", "true", "yes"),
            memory_dir=resolve_memory_dir(),
            memory_top_k=memory_top_k,
            skills_enabled=skills_raw in ("1", "true", "yes"),
            skills_dir=resolve_skill_dir(),
            skill_top_k=skill_top_k,
            skill_max_active=skill_max_active,
            skill_prompt_chars=skill_prompt_chars,
        )
