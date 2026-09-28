import socket

import pytest


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never call real services: no API key and no outbound sockets."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    def _blocked(*args: object, **kwargs: object) -> None:
        raise RuntimeError("network access is disabled in tests")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
