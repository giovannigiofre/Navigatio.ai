from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class TripState(TypedDict):
    """Shared state of one planning run."""

    messages: Annotated[list[BaseMessage], add_messages]

    # Routing decision written by the Supervisor and read by its conditional edge.
    next_agent: str
