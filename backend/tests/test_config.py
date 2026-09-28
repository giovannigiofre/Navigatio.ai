import pytest

from app.config import DEFAULT_ALLOWED_ORIGINS, Settings


def test_defaults_allow_only_localhost_origins(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)

    assert Settings.from_env().allowed_origins == DEFAULT_ALLOWED_ORIGINS
    assert all("localhost" in o or "127.0.0.1" in o for o in DEFAULT_ALLOWED_ORIGINS)


def test_allowed_origins_are_read_from_a_comma_separated_env_var(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("ALLOWED_ORIGINS", " https://a.example , https://b.example,,")

    assert Settings.from_env().allowed_origins == ("https://a.example", "https://b.example")


def test_max_steps_must_be_positive(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MAX_STEPS", "0")

    with pytest.raises(ValueError, match="MAX_STEPS"):
        Settings.from_env()
