import asyncio
from collections.abc import Iterator
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

from app.config import Settings, get_settings
from app.main import app, get_graph

SETTINGS = Settings(max_steps=4, max_history_turns=2, graph_timeout_s=5.0)


class FakeGraph:
    """Stands in for the compiled graph; records what the API passes to it."""

    def __init__(
        self, result: dict | None = None, error: Exception | None = None, delay: float = 0
    ):
        self.result = result
        self.error = error
        self.delay = delay
        self.inputs: dict | None = None
        self.config: dict | None = None

    async def ainvoke(self, inputs: dict, config: dict | None = None) -> dict:
        self.inputs, self.config = inputs, config
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return self.result or {}


FINAL_STATE = {
    "messages": [AIMessage("Your Lisbon weekend plan.")],
    "step_count": 1,
    "finish_reason": "supervisor",
    "trace": [
        {"node": "Supervisor", "latency_ms": 12.0, "decision": "WeatherAdvisor", "reason": "dates"},
        {
            "node": "WeatherAdvisor",
            "latency_ms": 40.5,
            "tool_calls": [{"tool": "get_trip_weather", "status": "ok", "days": 2}],
        },
        {"node": "Supervisor", "latency_ms": 9.0, "decision": "FINISH", "reason": "done"},
        {"node": "Finalize", "latency_ms": 20.0},
    ],
}


@pytest.fixture
def use() -> Iterator:
    """Install a FakeGraph (and optionally settings) for one test."""

    def _use(graph: FakeGraph, settings: Settings = SETTINGS) -> TestClient:
        app.dependency_overrides[get_graph] = lambda: graph
        app.dependency_overrides[get_settings] = lambda: settings
        return TestClient(app)  # no lifespan: nothing real is built

    yield _use
    app.dependency_overrides.clear()


def test_health(use):
    client = use(FakeGraph())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["X-Request-ID"]


def test_plan_returns_the_answer_and_a_structured_trace(use):
    client = use(FakeGraph(FINAL_STATE))

    response = client.post("/plan", json={"message": "Weekend in Lisbon, Oct 3-4"})

    assert response.status_code == 200
    body = response.json()
    assert body["response"] == "Your Lisbon weekend plan."
    assert body["steps"] == 1
    assert body["finish_reason"] == "supervisor"
    assert [step["node"] for step in body["trace"]] == [
        "Supervisor",
        "WeatherAdvisor",
        "Supervisor",
        "Finalize",
    ]
    assert body["trace"][1]["tool_calls"][0]["status"] == "ok"
    assert body["request_id"] == response.headers["X-Request-ID"]


def test_plan_passes_capped_history_step_budget_and_recursion_limit(use):
    graph = FakeGraph(FINAL_STATE)
    client = use(graph)
    history = [
        {"role": "user", "content": "Somewhere warm"},
        {"role": "assistant", "content": "Lisbon?"},
        {"role": "user", "content": "Yes, Lisbon"},
        {"role": "assistant", "content": "Which dates?"},
    ]

    client.post("/plan", json={"message": "Oct 3-4", "history": history})

    assert graph.inputs is not None
    # max_history_turns=2 keeps the last two turns; the conversation opens with a user turn.
    assert [m.content for m in graph.inputs["messages"]] == [
        "Yes, Lisbon",
        "Which dates?",
        "Oct 3-4",
    ]
    assert graph.inputs["max_steps"] == 4
    assert graph.inputs["step_count"] == 0
    assert graph.config == {"recursion_limit": 12}


def test_invalid_history_role_is_a_422_with_error_envelope(use):
    client = use(FakeGraph(FINAL_STATE))

    response = client.post(
        "/plan", json={"message": "hi", "history": [{"role": "system", "content": "x"}]}
    )

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["details"][0]["loc"][:3] == ["body", "history", 0]
    assert error["request_id"] == response.headers["X-Request-ID"]


def test_unexpected_errors_do_not_leak_internals(use):
    client = use(FakeGraph(error=RuntimeError("db password=hunter2 at /srv/app.py:42")))

    response = client.post("/plan", json={"message": "Weekend in Lisbon"})

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "hunter2" not in response.text
    assert "app.py" not in response.text


def test_llm_provider_errors_map_to_502(use):
    client = use(FakeGraph(error=ChatGoogleGenerativeAIError("429 quota exceeded for key AIza...")))

    response = client.post("/plan", json={"message": "Weekend in Lisbon"})

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "llm_unavailable"
    assert "AIza" not in response.text


def test_graph_run_is_bounded_by_a_timeout(use):
    client = use(FakeGraph(FINAL_STATE, delay=1.0), replace(SETTINGS, graph_timeout_s=0.05))

    response = client.post("/plan", json={"message": "Weekend in Lisbon"})

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "timeout"


def test_cors_preflight_allows_the_configured_origin(use):
    client = use(FakeGraph())

    allowed = client.options(
        "/plan",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
    )
    denied = client.options(
        "/plan",
        headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"},
    )

    assert allowed.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "access-control-allow-origin" not in denied.headers


def test_startup_fails_fast_without_an_api_key():
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"), TestClient(app):
        pass


def test_startup_builds_the_graph_and_closes_resources(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-used")

    with TestClient(app) as client:
        graph = client.app.state.graph
        assert set(graph.get_graph().nodes) >= {"Supervisor", "WeatherAdvisor", "Finalize"}
