"""Request and response models of the HTTP API."""

from typing import Literal

from pydantic import BaseModel, Field

MAX_MESSAGE_CHARS = 4_000
MAX_TURN_CHARS = 16_000
MAX_HISTORY_ITEMS = 200


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_TURN_CHARS)


class PlanRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    # Earlier turns, oldest first. Only the most recent MAX_HISTORY_TURNS are used.
    history: list[ChatTurn] = Field(default_factory=list, max_length=MAX_HISTORY_ITEMS)
