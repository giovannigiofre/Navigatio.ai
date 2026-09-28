from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.specialists import get_specialists
from app.agents.supervisor import FINISH, SPECIALISTS, create_supervisor_node
from app.config import Settings, get_settings
from app.graph.state import TripState


def route_from_supervisor(state: TripState) -> str:
    next_agent = state.get("next_agent", FINISH)
    return next_agent if next_agent in SPECIALISTS else FINISH


def build_graph(llm: BaseChatModel) -> CompiledStateGraph:
    """Supervisor/worker graph: Supervisor -> specialist -> Supervisor ... -> END."""
    workflow = StateGraph(TripState)
    workflow.add_node("Supervisor", create_supervisor_node(llm))
    for name, node in get_specialists(llm).items():
        workflow.add_node(name, node)
        workflow.add_edge(name, "Supervisor")

    workflow.add_edge(START, "Supervisor")
    workflow.add_conditional_edges(
        "Supervisor",
        route_from_supervisor,
        {**{name: name for name in SPECIALISTS}, FINISH: END},
    )
    return workflow.compile()


def create_workflow(settings: Settings | None = None) -> CompiledStateGraph:
    """Build the graph on Gemini. Requires GOOGLE_API_KEY in the environment."""
    settings = settings or get_settings()
    return build_graph(ChatGoogleGenerativeAI(model=settings.gemini_model))


def recursion_limit_for(max_steps: int) -> int:
    """LangGraph super-step limit matching a step budget, as a second safety net.

    One run is Supervisor + max_steps x (specialist + Supervisor), plus headroom.
    """
    return 2 * max_steps + 4


def initial_state(messages: Sequence[BaseMessage], max_steps: int) -> TripState:
    return {"messages": list(messages), "step_count": 0, "max_steps": max_steps}  # type: ignore[typeddict-item]


def run_config(max_steps: int) -> RunnableConfig:
    return {"recursion_limit": recursion_limit_for(max_steps)}
