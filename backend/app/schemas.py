"""Request and response models of the HTTP API."""

from typing import Literal

from pydantic import BaseModel, Field

MAX_MESSAGE_CHARS = 4_000
MAX_TURN_CHARS = 16_000
MAX_HISTORY_ITEMS = 50


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_TURN_CHARS)


class PlanRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    # Earlier turns, oldest first. Only the most recent MAX_HISTORY_TURNS are used.
    history: list[ChatTurn] = Field(default_factory=list, max_length=MAX_HISTORY_ITEMS)


class ToolCallTrace(BaseModel):
    tool: str
    status: Literal["ok", "error"]
    error_kind: str | None = None
    days: int | None = None


class TraceStep(BaseModel):
    node: str
    latency_ms: float
    decision: str | None = Field(default=None, description="Supervisor: next node or FINISH.")
    reason: str | None = Field(default=None, description="Supervisor: why.")
    tool_calls: list[ToolCallTrace] | None = None


class PlanResponse(BaseModel):
    request_id: str
    response: str
    steps: int = Field(description="Specialist steps taken in this run.")
    finish_reason: Literal["supervisor", "step_budget", "router_error"]
    trace: list[TraceStep]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str | None = None
    details: list[dict] | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
