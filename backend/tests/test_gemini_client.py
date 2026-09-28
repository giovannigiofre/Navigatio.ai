"""Run the graph on the real ChatGoogleGenerativeAI client against a mocked Gemini REST API.

The scripted fake model covers graph logic; this test covers the integration with
Gemini itself (request shapes and response parsing), still without network access.
"""

import json
from datetime import date

import httpx
import pytest
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.graph.workflow import build_graph, initial_state, run_config
from app.tools.trip_weather import TOOL_NAME, create_trip_weather_tool
from tests.fakes import StubWeatherProvider
from tests.test_weather_agent import FORECAST

GEMINI_URL = r"https://generativelanguage\.googleapis\.com/v1beta/models/[^/:]+:generateContent"


def candidate(*parts: dict) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "candidates": [
                {"content": {"role": "model", "parts": list(parts)}, "finishReason": "STOP"}
            ]
        },
    )


class FakeGemini:
    """Answers generateContent calls like Gemini would, and keeps the request bodies."""

    def __init__(self) -> None:
        self.bodies: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        system = body["systemInstruction"]["parts"][0]["text"]
        if "responseJsonSchema" in body.get("generationConfig", {}):  # supervisor routing
            first_turn = "Already consulted for this request: none." in system
            decision = "WeatherAdvisor" if first_turn else "FINISH"
            return candidate({"text": json.dumps({"next": decision, "reason": "test"})})
        if body.get("tools"):  # WeatherAdvisor choosing tool arguments
            args = {"destination": "Lisbon", "start_date": "2026-10-03", "end_date": "2026-10-03"}
            return candidate({"functionCall": {"name": TOOL_NAME, "args": args}})
        return candidate({"text": "Oct 3: moderate rain, 14-19.5 °C."})


async def test_graph_runs_on_the_gemini_client(respx_mock, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-used")
    gemini = FakeGemini()
    respx_mock.post(url__regex=GEMINI_URL).mock(side_effect=gemini)
    weather = StubWeatherProvider(FORECAST)
    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", timeout=5, max_retries=1)
    graph = build_graph(llm, create_trip_weather_tool(weather), today=lambda: date(2026, 9, 28))

    state = await graph.ainvoke(
        initial_state([HumanMessage("Weather in Lisbon on October 3?")], max_steps=6),
        config=run_config(6),
    )

    assert [(e["node"], e.get("decision")) for e in state["trace"]] == [
        ("Supervisor", "WeatherAdvisor"),
        ("WeatherAdvisor", None),
        ("Supervisor", "FINISH"),
        ("Finalize", None),
    ]
    assert weather.requests[0].start_date == date(2026, 10, 3)
    assert state["messages"][-1].content.startswith("Oct 3: moderate rain")

    routing = [b for b in gemini.bodies if "responseJsonSchema" in b.get("generationConfig", {})]
    assert len(routing) == 2
    route_schema = routing[0]["generationConfig"]["responseJsonSchema"]
    assert "FINISH" in route_schema["properties"]["next"]["enum"]

    [tool_request] = [b for b in gemini.bodies if b.get("tools")]
    declared = [d["name"] for t in tool_request["tools"] for d in t["functionDeclarations"]]
    assert declared == [TOOL_NAME]

    # Every request has a system instruction and a conversation that starts and ends
    # with the user's turn; specialist reports travel inside the system instruction.
    for body in gemini.bodies:
        assert body["contents"][0]["role"] == "user"
        assert body["contents"][-1]["role"] == "user"
