from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from app.agents.common import SpecialistName, build_llm_input
from app.graph.state import TripState


@dataclass(frozen=True)
class SimulatedSpecialist:
    name: SpecialistName
    prompt: str


FLIGHT_EXPERT = SimulatedSpecialist(
    name="FlightExpert",
    prompt="""You are an expert Flight Specialist. Today's date is {today}.
Your job is to find the best flight options based on the user's destination and dates.
For this prototype, simulate finding flights. Be specific about airlines, times, and prices.""",
)

HOTEL_SCOUT = SimulatedSpecialist(
    name="HotelScout",
    prompt="""You are a Hotel Scout. Today's date is {today}.
Your job is to find accommodation that matches the user's style and budget.
For this prototype, simulate finding hotels. Mention specific hotel names and why they fit.""",
)

ACTIVITY_CURATOR = SimulatedSpecialist(
    name="ActivityCurator",
    prompt="""You are an Activity Curator. Today's date is {today}.
Your job is to find cool things to do.
For this prototype, suggest a mix of tourist highlights and hidden gems.""",
)

SIMULATED_SPECIALISTS = (FLIGHT_EXPERT, HOTEL_SCOUT, ACTIVITY_CURATOR)


def create_specialist_node(
    llm: BaseChatModel, spec: SimulatedSpecialist, today: Callable[[], date] = date.today
):
    async def specialist(state: TripState) -> dict:
        prompt = spec.prompt.format(today=today().isoformat())
        response = await llm.ainvoke(build_llm_input(prompt, state["messages"]))
        return {"messages": [AIMessage(content=response.text, name=spec.name)]}

    return specialist


def get_specialists(llm: BaseChatModel, today: Callable[[], date] = date.today) -> dict:
    return {spec.name: create_specialist_node(llm, spec, today) for spec in SIMULATED_SPECIALISTS}
