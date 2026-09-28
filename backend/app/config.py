"""Runtime configuration, read once from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

# Stable (GA) Gemini Flash model. Override with GEMINI_MODEL; run
# `python scripts/list_models.py` to see what your API key can call.
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"


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

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            gemini_model=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
            max_steps=_int_env("MAX_STEPS", 6),
        )


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()
