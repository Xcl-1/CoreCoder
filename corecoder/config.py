"""Configuration - env vars and defaults."""

import os
import re
from dataclasses import dataclass
from pathlib import Path

_NAMESPACE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@-]{0,127}$")
_WINDOWS_RESERVED_NAMES = {
    "con", "prn", "aux", "nul",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}


def resolve_memory_dir(value: str | Path | None = None) -> Path:
    """Resolve the configured memory directory for CLI and library callers."""
    raw = value if value is not None else (os.getenv("CORECODER_MEMORY_DIR") or "~/.corecoder/memory")
    return Path(raw).expanduser().resolve()


def resolve_skill_dir(value: str | Path | None = None) -> Path:
    """Resolve the user-level skill directory without creating it."""
    raw = value if value is not None else (os.getenv("CORECODER_SKILLS_DIR") or "~/.corecoder/skills")
    return Path(raw).expanduser().resolve()


def resolve_context_artifacts_dir(value: str | Path | None = None) -> Path:
    """Resolve storage for externalized, session-scoped tool observations."""
    raw = value if value is not None else (
        os.getenv("CORECODER_CONTEXT_ARTIFACTS_DIR") or "~/.corecoder/context-artifacts"
    )
    return Path(raw).expanduser().resolve()


def validate_namespace(value: str, field: str) -> str:
    """Validate an externally supplied tenant or user identifier for path use."""
    normalized = value.strip()
    if not normalized:
        return ""
    canonical = normalized.casefold()
    if (
        canonical in {".", ".."}
        or canonical.rstrip(".") in _WINDOWS_RESERVED_NAMES
        or normalized.endswith(".")
        or not _NAMESPACE_RE.fullmatch(normalized)
    ):
        raise ValueError(
            f"{field} must be 1-128 characters using letters, numbers, '.', '_', '@', or '-'"
        )
    return canonical


def scoped_data_dir(
    root: str | Path,
    *,
    tenant_id: str = "",
    user_id: str = "",
) -> Path:
    """Return a traversal-safe per-tenant/per-user data directory.

    Empty identifiers preserve the historical single-user layout. Deployments
    can opt into isolation without migrating existing local installations.
    """
    path = Path(root).expanduser().resolve()
    tenant = validate_namespace(tenant_id, "tenant_id")
    user = validate_namespace(user_id, "user_id")
    if tenant:
        path /= Path("tenants") / tenant
    if user:
        path /= Path("users") / user
    return path.resolve()


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
    skill_max_active: int = 3
    skill_prompt_chars: int = 6_000
    skill_min_score: float = 0.24
    skill_auto_confidence: float = 0.82
    skill_clarify_confidence: float = 0.65
    skill_ambiguity_margin: float = 0.12
    context_artifacts_enabled: bool = True
    context_artifacts_dir: Path = Path.home() / ".corecoder" / "context-artifacts"
    context_artifact_threshold: int = 12_000
    context_artifact_ttl_days: int = 30
    context_artifact_max_mb: int = 256
    tenant_id: str = ""
    user_id: str = ""

    @property
    def memory_data_dir(self) -> Path:
        return scoped_data_dir(
            self.memory_dir,
            tenant_id=self.tenant_id,
            user_id=self.user_id,
        )

    @property
    def skills_data_dir(self) -> Path:
        return scoped_data_dir(
            self.skills_dir,
            tenant_id=self.tenant_id,
            user_id=self.user_id,
        )

    @property
    def context_artifacts_data_dir(self) -> Path:
        return scoped_data_dir(
            self.context_artifacts_dir,
            tenant_id=self.tenant_id,
            user_id=self.user_id,
        )

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
        skill_max_active_raw = os.getenv("CORECODER_SKILL_MAX_ACTIVE", "3")
        skill_prompt_chars_raw = os.getenv("CORECODER_SKILL_PROMPT_CHARS", "6000")
        skill_min_score_raw = os.getenv("CORECODER_SKILL_MIN_SCORE", "0.24")
        skill_auto_confidence_raw = os.getenv("CORECODER_SKILL_AUTO_CONFIDENCE", "0.82")
        skill_clarify_confidence_raw = os.getenv("CORECODER_SKILL_CLARIFY_CONFIDENCE", "0.65")
        skill_ambiguity_margin_raw = os.getenv("CORECODER_SKILL_AMBIGUITY_MARGIN", "0.12")
        context_artifacts_raw = os.getenv("CORECODER_CONTEXT_ARTIFACTS", "1").strip().lower()
        context_artifact_threshold_raw = os.getenv(
            "CORECODER_CONTEXT_ARTIFACT_THRESHOLD", "12000"
        )
        context_artifact_ttl_days_raw = os.getenv("CORECODER_CONTEXT_ARTIFACT_TTL_DAYS", "30")
        context_artifact_max_mb_raw = os.getenv("CORECODER_CONTEXT_ARTIFACT_MAX_MB", "256")
        tenant_id = validate_namespace(os.getenv("CORECODER_TENANT_ID", ""), "CORECODER_TENANT_ID")
        user_id = validate_namespace(os.getenv("CORECODER_USER_ID", ""), "CORECODER_USER_ID")

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
        if context_artifacts_raw not in ("1", "true", "yes", "0", "false", "no"):
            raise ValueError(
                "CORECODER_CONTEXT_ARTIFACTS must be a boolean, "
                f"got: {context_artifacts_raw!r}"
            )
        try:
            context_artifact_threshold = int(context_artifact_threshold_raw)
        except ValueError as exc:
            raise ValueError(
                "CORECODER_CONTEXT_ARTIFACT_THRESHOLD must be an integer, "
                f"got: {context_artifact_threshold_raw!r}"
            ) from exc
        if not (1_000 <= context_artifact_threshold <= 1_000_000):
            raise ValueError(
                "CORECODER_CONTEXT_ARTIFACT_THRESHOLD must be 1000-1000000, "
                f"got: {context_artifact_threshold}"
            )
        try:
            context_artifact_ttl_days = int(context_artifact_ttl_days_raw)
            context_artifact_max_mb = int(context_artifact_max_mb_raw)
        except ValueError as exc:
            raise ValueError("Context artifact TTL and capacity must be integers") from exc
        if not (1 <= context_artifact_ttl_days <= 3_650):
            raise ValueError(
                "CORECODER_CONTEXT_ARTIFACT_TTL_DAYS must be 1-3650, "
                f"got: {context_artifact_ttl_days}"
            )
        if not (1 <= context_artifact_max_mb <= 102_400):
            raise ValueError(
                "CORECODER_CONTEXT_ARTIFACT_MAX_MB must be 1-102400, "
                f"got: {context_artifact_max_mb}"
            )
        try:
            skill_top_k = int(skill_top_k_raw)
            skill_max_active = int(skill_max_active_raw)
            skill_prompt_chars = int(skill_prompt_chars_raw)
        except ValueError as exc:
            raise ValueError("Skill limits must be integers") from exc
        try:
            skill_min_score = float(skill_min_score_raw)
            skill_auto_confidence = float(skill_auto_confidence_raw)
            skill_clarify_confidence = float(skill_clarify_confidence_raw)
            skill_ambiguity_margin = float(skill_ambiguity_margin_raw)
        except ValueError as exc:
            raise ValueError("Skill routing thresholds must be numbers") from exc
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
        if not (0 <= skill_min_score <= 2):
            raise ValueError(
                f"CORECODER_SKILL_MIN_SCORE must be 0-2, got: {skill_min_score}"
            )
        if not (0 <= skill_ambiguity_margin <= 1):
            raise ValueError(
                "CORECODER_SKILL_AMBIGUITY_MARGIN must be 0-1, "
                f"got: {skill_ambiguity_margin}"
            )
        if not (0 <= skill_clarify_confidence <= skill_auto_confidence <= 1):
            raise ValueError(
                "Skill confidence thresholds must satisfy 0 <= clarify <= auto <= 1, "
                f"got clarify={skill_clarify_confidence}, auto={skill_auto_confidence}"
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
            skill_min_score=skill_min_score,
            skill_auto_confidence=skill_auto_confidence,
            skill_clarify_confidence=skill_clarify_confidence,
            skill_ambiguity_margin=skill_ambiguity_margin,
            context_artifacts_enabled=context_artifacts_raw in ("1", "true", "yes"),
            context_artifacts_dir=resolve_context_artifacts_dir(),
            context_artifact_threshold=context_artifact_threshold,
            context_artifact_ttl_days=context_artifact_ttl_days,
            context_artifact_max_mb=context_artifact_max_mb,
            tenant_id=tenant_id,
            user_id=user_id,
        )
