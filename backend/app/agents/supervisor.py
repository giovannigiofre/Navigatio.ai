import logging
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from app.agents.common import FINISH, SpecialistName, build_llm_input, specialist_reports
from app.graph.state import TripState

logger = logging.getLogger(__name__)

DEFAULT_MAX_STEPS = 6

SYSTEM_PROMPT = """You are the supervisor of a trip-planning team. You do not write the plan \
yourself: you decide which specialist acts next, or FINISH.

Specialists:
- FlightExpert: flight options between the user's origin and destination.
- HotelScout: where to stay, matched to the user's style and budget.
- ActivityCurator: things to do and places to eat at the destination.

Rules:
- Call only the specialists the request actually needs. "Things to do in Lisbon" needs \
ActivityCurator only.
- Call each specialist at most once per request unless its report was clearly incomplete.
- Choose FINISH when the specialists' reports cover the request.
- Choose FINISH right away when key information is missing (for example no destination, or no \
dates for flights and hotels): the user will be asked a clarifying question.

Specialist steps used so far: {step_count} of {max_steps}.
Already consulted for this request: {consulted}."""


class RouteDecision(BaseModel):
    """The supervisor's choice of what happens next."""

    next: Literal[SpecialistName, "FINISH"] = Field(
        description="The specialist that should act next, or FINISH to end the run."
    )
    reason: str = Field(description="One short sentence explaining the choice.")


def _router_input(state: TripState, step_count: int, max_steps: int) -> list[BaseMessage]:
    consulted = [m.name for m in specialist_reports(state["messages"])]
    prompt = SYSTEM_PROMPT.format(
        step_count=step_count,
        max_steps=max_steps,
        consulted=", ".join(consulted) if consulted else "none",
    )
    return build_llm_input(prompt, state["messages"])


def create_supervisor_node(llm: BaseChatModel):
    router = llm.with_structured_output(RouteDecision, include_raw=True)

    async def supervisor(state: TripState) -> dict:
        step_count = state.get("step_count", 0)
        max_steps = state.get("max_steps", DEFAULT_MAX_STEPS)

        if step_count >= max_steps:
            logger.warning(
                "step budget exhausted, forcing FINISH",
                extra={"step_count": step_count, "max_steps": max_steps},
            )
            return {"next_agent": FINISH, "finish_reason": "step_budget"}

        result = await router.ainvoke(_router_input(state, step_count, max_steps))
        decision: RouteDecision | None = result["parsed"]
        if decision is None:
            logger.error(
                "supervisor returned no valid route, forcing FINISH",
                extra={"parsing_error": repr(result["parsing_error"])},
            )
            return {"next_agent": FINISH, "finish_reason": "router_error"}

        logger.info(
            "route decision",
            extra={
                "next": decision.next,
                "reason": decision.reason,
                "step_count": step_count,
                "max_steps": max_steps,
            },
        )
        if decision.next == FINISH:
            return {"next_agent": FINISH, "finish_reason": "supervisor"}
        return {"next_agent": decision.next, "step_count": step_count + 1}

    return supervisor
