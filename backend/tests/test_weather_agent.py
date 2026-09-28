from datetime import date

from langchain_core.messages import AIMessage, HumanMessage

from app.agents.common import FINISH
from app.agents.weather import LIVE_SOURCE_NOTE, UNAVAILABLE_SOURCE_NOTE
from app.tools.models import DailyWeather, Place, ToolError, TripWeather
from app.tools.trip_weather import TOOL_NAME
from tests.fakes import ScriptedChatModel, StubWeatherProvider, reply, route, tool_call
from tests.test_graph import run

LISBON = Place(name="Lisbon, Portugal", latitude=38.71, longitude=-9.14, country_code="pt")
FORECAST = TripWeather(
    place=LISBON,
    days=[
        DailyWeather(
            date=date(2026, 10, 3),
            summary="moderate rain",
            temperature_max_c=19.5,
            temperature_min_c=14.0,
            precipitation_mm=6.1,
            precipitation_probability_pct=80,
        )
    ],
)
REQUEST = [HumanMessage("Weekend in Lisbon, October 3 to 4. What should I do?")]


def weather_report(state: dict) -> AIMessage:
    return next(m for m in state["messages"] if getattr(m, "name", None) == "WeatherAdvisor")


async def test_forecast_is_fetched_summarised_and_used_by_activities():
    weather = StubWeatherProvider(FORECAST)
    llm = ScriptedChatModel(
        script=[
            route("WeatherAdvisor"),
            tool_call(
                TOOL_NAME, destination="Lisbon", start_date="2026-10-03", end_date="2026-10-04"
            ),
            reply("Oct 3: moderate rain, 14-19.5 °C, 80% rain. Plan indoor activities."),
            route("ActivityCurator"),
            reply("Rainy day: Gulbenkian Museum."),
            route(FINISH),
            reply("Your weekend."),
        ]
    )

    state = await run(llm, REQUEST, weather=weather)

    [request] = weather.requests
    assert (request.destination, request.start_date, request.end_date) == (
        "Lisbon",
        date(2026, 10, 3),
        date(2026, 10, 4),
    )
    assert weather_report(state).content.startswith("Oct 3: moderate rain")
    # The summary step sees only validated tool output, as JSON.
    summary_call = llm.calls[2]
    assert '"summary":"moderate rain"' in summary_call.system_prompt
    assert summary_call.tools == []
    # ActivityCurator gets the weather report to choose indoor vs outdoor plans.
    assert "### WeatherAdvisor" in llm.calls[4].system_prompt
    assert f"- {LIVE_SOURCE_NOTE}" in state["messages"][-1].content


async def test_tool_failure_is_reported_without_asking_the_model_about_weather():
    weather = StubWeatherProvider(
        ToolError(kind="upstream_unavailable", message="The Open-Meteo request failed (HTTP 503).")
    )
    llm = ScriptedChatModel(
        script=[
            route("WeatherAdvisor"),
            tool_call(
                TOOL_NAME, destination="Lisbon", start_date="2026-10-03", end_date="2026-10-04"
            ),
            route(FINISH),
            reply("Weather is unavailable right now."),
        ]
    )

    state = await run(llm, REQUEST, weather=weather)

    report = weather_report(state).content
    assert report.startswith("Weather forecast unavailable.")
    assert "HTTP 503" in report
    assert len(llm.calls) == 4  # no summary call after the failed tool call
    assert f"- {UNAVAILABLE_SOURCE_NOTE}" in state["messages"][-1].content
    assert LIVE_SOURCE_NOTE not in state["messages"][-1].content


async def test_invalid_tool_arguments_become_a_structured_error():
    weather = StubWeatherProvider()
    llm = ScriptedChatModel(
        script=[
            route("WeatherAdvisor"),
            tool_call(
                TOOL_NAME, destination="Lisbon", start_date="2026-10-04", end_date="2026-10-03"
            ),
            route(FINISH),
            reply("ok"),
        ]
    )

    state = await run(llm, REQUEST, weather=weather)

    assert weather.requests == []
    assert "Invalid tool arguments" in weather_report(state).content


async def test_missing_dates_means_no_tool_call():
    weather = StubWeatherProvider()
    llm = ScriptedChatModel(
        script=[
            route("WeatherAdvisor"),
            reply("I need the travel dates to check the forecast."),
            route(FINISH),
            reply("When are you travelling?"),
        ]
    )

    state = await run(llm, [HumanMessage("Weather in Lisbon for my trip?")], weather=weather)

    assert weather.requests == []
    assert weather_report(state).content == "I need the travel dates to check the forecast."
    assert "Data sources" not in state["messages"][-1].content
