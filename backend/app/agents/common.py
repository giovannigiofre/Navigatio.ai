"""Names and message helpers shared by the supervisor and the specialists."""

from collections.abc import Sequence
from typing import Literal, get_args

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage

SpecialistName = Literal["FlightExpert", "HotelScout", "ActivityCurator", "WeatherAdvisor"]
SPECIALISTS: tuple[str, ...] = get_args(SpecialistName)
FINISH = "FINISH"


def is_specialist_report(message: BaseMessage) -> bool:
    return isinstance(message, AIMessage) and message.name in SPECIALISTS


def specialist_reports(messages: Sequence[BaseMessage]) -> list[AIMessage]:
    return [m for m in messages if is_specialist_report(m)]  # type: ignore[misc]


def build_llm_input(system_prompt: str, messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """System prompt + the user-facing conversation, with specialist reports attributed.

    Specialist reports are moved into the system prompt under their author's name,
    so the model knows who wrote what and the message list still ends with the
    user's turn. The full history is kept on purpose: it is short (capped by the
    API) and every agent needs the user's constraints.
    """
    conversation = [m for m in messages if not is_specialist_report(m)]
    reports = specialist_reports(messages)
    if reports:
        rendered = "\n\n".join(f"### {m.name}\n{m.text}" for m in reports)
        system_prompt = (
            f"{system_prompt}\n\nReports from specialists for the current request:\n\n{rendered}"
        )
    return [SystemMessage(content=system_prompt), *conversation]
