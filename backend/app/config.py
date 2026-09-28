"""Runtime configuration, read once from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

# Stable (GA) Gemini Flash model. Override with GEMINI_MODEL; run
# `python scripts/list_models.py` to see what your API key can call.
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"

# Nominatim's usage policy requires a User-Agent that identifies the application.
DEFAULT_HTTP_USER_AGENT = "Navigatio.ai/0.1 (+https://github.com/giovannigiofre/Navigatio.ai)"


DEFAULT_ALLOWED_ORIGINS = ("http://localhost:3000", "http://127.0.0.1:3000")


def _list_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _int_env(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    value = int(raw)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    gemini_model: str = DEFAULT_GEMINI_MODEL
    # Maximum specialist delegations per request before the supervisor is forced to FINISH.
    max_steps: int = 6
    # Earlier chat turns passed to the graph with each request (older ones are dropped).
    max_history_turns: int = 20
    # Sent with every external API call.
    http_user_agent: str = DEFAULT_HTTP_USER_AGENT
    log_level: str = "INFO"
    # Browser origins allowed to call the API (CORS).
    allowed_origins: tuple[str, ...] = DEFAULT_ALLOWED_ORIGINS

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            gemini_model=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
            max_steps=_int_env("MAX_STEPS", 6),
            max_history_turns=_int_env("MAX_HISTORY_TURNS", 20, minimum=0),
            http_user_agent=os.getenv("HTTP_USER_AGENT") or DEFAULT_HTTP_USER_AGENT,
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            allowed_origins=_list_env("ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS),
        )


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()
