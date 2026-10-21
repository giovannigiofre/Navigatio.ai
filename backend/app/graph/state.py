import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class TraceEvent(TypedDict, total=False):
    """One node execution, in order. Returned by the API for debugging and the UI."""

    node: str
    latency_ms: float
    decision: str  # Supervisor: next specialist or FINISH
    reason: str  # Supervisor: why
    tool_calls: list[dict[str, Any]]  # WeatherAdvisor: tool name, status, error kind


class TripState(TypedDict):
    """Shared state of one planning run."""

    messages: Annotated[list[BaseMessage], add_messages]

    # Routing decision written by the Supervisor and read by its conditional edge.
    next_agent: str

    # Step budget: number of specialist delegations so far, and the maximum allowed.
    # The Supervisor forces FINISH once step_count reaches max_steps.
    step_count: int
    max_steps: int

    # Why the run ended: "supervisor" (chose FINISH), "step_budget" or "router_error".
    finish_reason: str

    # Provenance notes from the specialists that ran (e.g. "not live data"),
    # appended verbatim to the final answer.
    sources: Annotated[list[str], operator.add]

    # Per-node trace, appended by the workflow's node wrapper.
    trace: Annotated[list[TraceEvent], operator.add]
