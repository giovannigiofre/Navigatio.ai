"""Runtime configuration, read once from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

# Stable (GA) Gemini Flash model. Override with GEMINI_MODEL; run
# `python scripts/list_models.py` to see what your API key can call.
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"


@dataclass(frozen=True)
class Settings:
    gemini_model: str = DEFAULT_GEMINI_MODEL

    @classmethod
    def from_env(cls) -> Settings:
        return cls(gemini_model=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL))


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()
