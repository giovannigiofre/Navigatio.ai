"""A scripted chat model so graph logic can be tested without calling an LLM."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

_ids = itertools.count()


@dataclass
class Call:
    messages: list[BaseMessage]
    tools: list[str]

    @property
    def system_prompt(self) -> str:
        first = self.messages[0]
        return first.content if isinstance(first, SystemMessage) else ""  # type: ignore[return-value]

    @property
    def is_route(self) -> bool:
        return "RouteDecision" in self.tools


Responder = Callable[[Call], AIMessage]


class ScriptedChatModel(BaseChatModel):
    """Returns scripted replies and records every call.

    `script` is either a list of replies consumed in order, or a function that
    builds the reply from the call (useful for open-ended loops).
    """

    script: list[AIMessage] | Responder
    calls: list[Call] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any):  # type: ignore[override]
        names = [convert_to_openai_tool(tool)["function"]["name"] for tool in tools]
        return self.bind(bound_tools=names)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        bound_tools: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        call = Call(messages=list(messages), tools=list(bound_tools or []))
        self.calls.append(call)
        if callable(self.script):
            reply = self.script(call)
        else:
            if not self.script:
                raise AssertionError(f"script exhausted; unexpected call #{len(self.calls)}")
            reply = self.script.pop(0)
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def route_calls(self) -> list[Call]:
        return [call for call in self.calls if call.is_route]


def route(next_agent: str, reason: str = "scripted") -> AIMessage:
    """A supervisor reply choosing `next_agent` (a specialist name or FINISH)."""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "RouteDecision",
                "args": {"next": next_agent, "reason": reason},
                "id": f"route-{next(_ids)}",
            }
        ],
    )


def reply(text: str) -> AIMessage:
    return AIMessage(content=text)


def tool_call(name: str, **args: Any) -> AIMessage:
    """A model reply that calls tool `name` with `args`."""
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": f"call-{next(_ids)}"}]
    )


class StubWeatherProvider:
    """Stands in for TripWeatherService: returns canned results, records requests."""

    def __init__(self, *results: Any) -> None:
        self._results = list(results)
        self.requests: list[Any] = []

    async def forecast(self, request: Any) -> Any:
        self.requests.append(request)
        return self._results.pop(0)
