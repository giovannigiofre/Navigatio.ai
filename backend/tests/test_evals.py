from datetime import date

import pytest

from app.agents.common import SPECIALISTS
from app.config import Settings
from app.graph.workflow import build_graph
from app.tools.trip_weather import create_trip_weather_tool
from evals.run_routing_eval import (
    DEFAULT_CASES,
    Case,
    load_cases,
    precision_recall,
    print_report,
    run_case,
)
from tests.fakes import ScriptedChatModel, StubWeatherProvider, reply, route


def f(*names: str) -> frozenset[str]:
    return frozenset(names)


@pytest.mark.parametrize(
    ("expected", "called", "scores"),
    [
        (f("FlightExpert", "HotelScout"), f("FlightExpert", "HotelScout"), (1.0, 1.0)),
        (f("FlightExpert", "HotelScout"), f("FlightExpert"), (1.0, 0.5)),
        (f("ActivityCurator"), f("ActivityCurator", "WeatherAdvisor"), (0.5, 1.0)),
        (f(), f(), (1.0, 1.0)),  # clarifying question: nothing expected, nothing called
        (f(), f("HotelScout"), (0.0, 1.0)),
        (f("HotelScout"), f(), (1.0, 0.0)),
    ],
)
def test_precision_recall(expected, called, scores):
    assert precision_recall(expected, called) == scores


def test_routing_cases_are_well_formed():
    cases = load_cases(DEFAULT_CASES)

    assert 12 <= len(cases) <= 15
    assert len({case.id for case in cases}) == len(cases)
    assert all(case.expected <= set(SPECIALISTS) for case in cases)
    assert any(not case.expected for case in cases), "needs a clarifying-question case"
    assert any(case.history for case in cases), "needs a multi-turn case"


async def test_run_case_scores_the_specialists_the_graph_called(capsys):
    llm = ScriptedChatModel(
        script=[route("FlightExpert"), reply("Flights."), route("FINISH"), reply("Plan.")]
    )
    graph = build_graph(
        llm, create_trip_weather_tool(StubWeatherProvider()), today=lambda: date(2026, 9, 28)
    )
    case = Case(
        id="c1", message="Flights and a hotel in Rome", expected=f("FlightExpert", "HotelScout")
    )

    result = await run_case(graph, case, Settings())
    print_report([result], Settings())

    assert result.called == f("FlightExpert")
    assert (result.precision, result.recall) == (1.0, 0.5)
    assert result.steps == 1
    assert result.error is None
    assert "mean recall:      0.500" in capsys.readouterr().out
