import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from datetime import date

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.common import FINISH, SPECIALISTS
from app.agents.finalizer import create_finalizer_node
from app.agents.specialists import get_specialists
from app.agents.supervisor import create_supervisor_node
from app.agents.weather import NAME as WEATHER_ADVISOR
from app.agents.weather import create_weather_advisor_node
from app.config import Settings, get_settings
from app.graph.state import TripState
from app.tools.trip_weather import build_trip_weather_service, create_trip_weather_tool

logger = logging.getLogger(__name__)

LLM_MAX_ATTEMPTS = 2

Node = Callable[[TripState], Awaitable[dict]]


def traced(name: str, node: Node) -> Node:
    """Time a node, log it, and append a TraceEvent built from its optional `trace_detail`."""

    async def wrapper(state: TripState) -> dict:
        started = time.perf_counter()
        try:
            update = await node(state)
        except Exception as exc:
            latency_ms = round((time.perf_counter() - started) * 1000, 1)
            logger.warning(
                "node failed",
                extra={"node": name, "latency_ms": latency_ms, "error": type(exc).__name__},
            )
            raise
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        detail = update.pop("trace_detail", {})
        logger.info(
            "node complete",
            extra={
                "node": name,
                "latency_ms": latency_ms,
                "step_count": update.get("step_count", state.get("step_count", 0)),
                **detail,
            },
        )
        return {**update, "trace": [{"node": name, "latency_ms": latency_ms, **detail}]}

    return wrapper


def route_from_supervisor(state: TripState) -> str:
    next_agent = state.get("next_agent", FINISH)
    return next_agent if next_agent in SPECIALISTS else FINISH


def build_graph(
    llm: BaseChatModel, weather_tool: BaseTool, today: Callable[[], date] = date.today
) -> CompiledStateGraph:
    """Supervisor -> specialist -> Supervisor ... -> Finalize -> END."""
    specialists = {
        **get_specialists(llm, today),
        WEATHER_ADVISOR: create_weather_advisor_node(llm, weather_tool, today),
    }
    workflow = StateGraph(TripState)
    workflow.add_node("Supervisor", traced("Supervisor", create_supervisor_node(llm, today)))
    for name, node in specialists.items():
        workflow.add_node(name, traced(name, node))
        workflow.add_edge(name, "Supervisor")
    workflow.add_node("Finalize", traced("Finalize", create_finalizer_node(llm)))

    workflow.add_edge(START, "Supervisor")
    workflow.add_conditional_edges(
        "Supervisor",
        route_from_supervisor,
        {**{name: name for name in SPECIALISTS}, FINISH: "Finalize"},
    )
    workflow.add_edge("Finalize", END)
    return workflow.compile()


def create_workflow(
    http: httpx.AsyncClient, settings: Settings | None = None
) -> CompiledStateGraph:
    """Build the graph on Gemini. Requires GOOGLE_API_KEY in the environment.

    `http` is the shared client for the weather tool; its owner closes it.
    """
    settings = settings or get_settings()
    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        timeout=settings.llm_timeout_s,
        # google-genai treats this as total attempts; it retries 408/429/5xx with backoff.
        max_retries=LLM_MAX_ATTEMPTS,
    )
    weather_tool = create_trip_weather_tool(build_trip_weather_service(http))
    return build_graph(llm, weather_tool)


def recursion_limit_for(max_steps: int) -> int:
    """LangGraph super-step limit matching a step budget, as a second safety net.

    One run is Supervisor + max_steps x (specialist + Supervisor) + Finalize, plus headroom.
    """
    return 2 * max_steps + 4


def initial_state(messages: Sequence[BaseMessage], max_steps: int) -> TripState:
    return {"messages": list(messages), "step_count": 0, "max_steps": max_steps}  # type: ignore[typeddict-item]


def run_config(max_steps: int) -> RunnableConfig:
    return {"recursion_limit": recursion_limit_for(max_steps)}
