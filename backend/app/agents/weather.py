"""WeatherAdvisor: the one specialist backed by live data, through get_trip_weather.

Flow: the model turns the conversation into tool arguments (a tool call), the
tool runs in code, and the model then summarises the validated result. If
every tool call fails, the report is a fixed message and the model is not
asked to write about weather it does not have.
"""

import logging
from collections.abc import Callable
from datetime import date

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolCall
from langchain_core.tools import BaseTool
from pydantic import ValidationError

from app.agents.common import build_llm_input
from app.graph.state import TripState
from app.tools.models import ToolError, TripWeatherResult

logger = logging.getLogger(__name__)

NAME = "WeatherAdvisor"
MAX_TOOL_CALLS = 3

EXTRACT_PROMPT = """You are the WeatherAdvisor on a trip-planning team. Today's date is {today}.
Call get_trip_weather with the trip's destination and dates (YYYY-MM-DD). Resolve relative \
dates such as "next weekend" from today's date. Use one call per destination.
If the destination or the dates are unknown, do not call the tool: reply in one sentence \
saying what is missing. Never state weather facts yourself."""

SUMMARY_PROMPT = """You are the WeatherAdvisor on a trip-planning team. Summarise the forecast \
data below for the trip: one line per day with conditions, min/max temperature in °C and \
chance of rain, then say which days suit outdoor plans and which call for indoor ones.
Use only numbers from the data. If a result has "status": "error", say plainly that this \
forecast is unavailable and why. Plain text, under 120 words.

Forecast data from get_trip_weather:
{data}"""

LIVE_SOURCE_NOTE = (
    "Weather is a live forecast from Open-Meteo (place lookup by OpenStreetMap Nominatim)."
)
UNAVAILABLE_SOURCE_NOTE = "Weather: no live forecast was available for this request."


async def run_weather_tool(tool: BaseTool, call: ToolCall) -> TripWeatherResult:
    try:
        message = await tool.ainvoke({**call, "type": "tool_call"})
    except ValidationError as exc:
        logger.warning(
            "invalid tool arguments", extra={"tool": tool.name, "tool_args": call["args"]}
        )
        fields = ", ".join(str(err["loc"][0]) for err in exc.errors() if err["loc"]) or "input"
        return ToolError(kind="invalid_input", message=f"Invalid tool arguments ({fields}).")
    return message.artifact


def create_weather_advisor_node(
    llm: BaseChatModel, weather_tool: BaseTool, today: Callable[[], date] = date.today
):
    llm_with_tool = llm.bind_tools([weather_tool])

    async def weather_advisor(state: TripState) -> dict:
        messages = state["messages"]
        prompt = EXTRACT_PROMPT.format(today=today().isoformat())
        request = await llm_with_tool.ainvoke(build_llm_input(prompt, messages))

        calls = request.tool_calls[:MAX_TOOL_CALLS]
        if len(request.tool_calls) > MAX_TOOL_CALLS:
            logger.warning(
                "too many tool calls, extra calls ignored",
                extra={"requested": len(request.tool_calls), "max": MAX_TOOL_CALLS},
            )
        if not calls:
            text = request.text or "No forecast: the destination or dates are missing."
            return {"messages": [AIMessage(content=text, name=NAME)]}

        results = [await run_weather_tool(weather_tool, call) for call in calls]
        errors = [r for r in results if isinstance(r, ToolError)]
        if len(errors) == len(results):
            reasons = " ".join(error.message for error in errors)
            text = f"Weather forecast unavailable. {reasons} No forecast is included in this plan."
            return {
                "messages": [AIMessage(content=text, name=NAME)],
                "sources": [UNAVAILABLE_SOURCE_NOTE],
            }

        data = "\n".join(result.model_dump_json() for result in results)
        summary = await llm.ainvoke(build_llm_input(SUMMARY_PROMPT.format(data=data), messages))
        return {
            "messages": [AIMessage(content=summary.text, name=NAME)],
            "sources": [LIVE_SOURCE_NOTE],
        }

    return weather_advisor
