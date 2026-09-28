"""FastAPI app: POST /plan runs the agent graph, GET /health reports liveness."""

import asyncio
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google.genai.errors import APIError as GenAIAPIError
from langchain_core.exceptions import ModelError
from langchain_core.runnables import Runnable
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

# Load backend/.env before any setting is read.
load_dotenv()

from app.config import Settings, get_settings  # noqa: E402
from app.conversation import build_conversation  # noqa: E402
from app.graph.workflow import create_workflow, initial_state, run_config  # noqa: E402
from app.logging_config import configure_logging, request_id_var  # noqa: E402
from app.schemas import (  # noqa: E402
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    PlanRequest,
    PlanResponse,
    TraceStep,
)
from app.tools.http import create_http_client  # noqa: E402

configure_logging(get_settings().log_level)
logger = logging.getLogger("app.api")

# Errors raised by the Gemini client (auth, quota, invalid request, 5xx after its retries).
LLM_PROVIDER_ERRORS = (ChatGoogleGenerativeAIError, GenAIAPIError, ModelError)


class ApiError(Exception):
    """An error returned to the client as {"error": {...}} with a safe, generic message."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create the shared HTTP client and the graph once; close the client on shutdown."""
    settings = get_settings()
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        raise RuntimeError(
            "GOOGLE_API_KEY is not set: copy backend/.env.example to backend/.env and add a key."
        )
    async with create_http_client(settings.http_user_agent) as http:
        app.state.graph = create_workflow(http, settings)
        logger.info(
            "startup complete",
            extra={"model": settings.gemini_model, "max_steps": settings.max_steps},
        )
        yield


app = FastAPI(title="Navigatio.ai Agent Backend", lifespan=lifespan)

_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Tag the request (and every log line it produces) with an id, and log its latency."""
    incoming = request.headers.get("X-Request-ID", "")
    request_id = incoming if _REQUEST_ID.fullmatch(incoming) else uuid.uuid4().hex
    token = request_id_var.set(request_id)
    started = time.perf_counter()
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request complete",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        return response
    finally:
        request_id_var.reset(token)


# Added last, so it is the outermost middleware and also answers preflight requests.
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(get_settings().allowed_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
    max_age=3600,
)


def _error_response(
    status_code: int, code: str, message: str, details: list[dict] | None = None
) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorDetail(
            code=code, message=message, request_id=request_id_var.get(), details=details
        )
    )
    return JSONResponse(status_code=status_code, content=body.model_dump(exclude_none=True))


@app.exception_handler(ApiError)
async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
    return _error_response(exc.status_code, exc.code, exc.message)


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Report where and why, without echoing the submitted values back.
    details = [{"loc": list(err["loc"]), "msg": err["msg"]} for err in exc.errors()]
    return _error_response(422, "invalid_request", "The request is invalid.", details)


def get_graph(request: Request) -> Runnable:
    return request.app.state.graph


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness: the process is up and serving requests."""
    return HealthResponse()


@app.post(
    "/plan",
    response_model=PlanResponse,
    responses={
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
)
async def plan_trip(
    body: PlanRequest,
    graph: Annotated[Runnable, Depends(get_graph)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> PlanResponse:
    messages = build_conversation(body.history, body.message, settings.max_history_turns)
    started = time.perf_counter()
    try:
        async with asyncio.timeout(settings.graph_timeout_s):
            state = await graph.ainvoke(
                initial_state(messages, settings.max_steps),
                config=run_config(settings.max_steps),
            )
    except TimeoutError as exc:
        logger.error("plan timed out", extra={"timeout_s": settings.graph_timeout_s})
        raise ApiError(504, "timeout", "Planning took too long. Please try again.") from exc
    except LLM_PROVIDER_ERRORS as exc:
        logger.exception("language model error")
        raise ApiError(
            502, "llm_unavailable", "The language model is unavailable. Please try again later."
        ) from exc
    except Exception as exc:
        logger.exception("plan failed")
        raise ApiError(
            500, "internal_error", "Something went wrong while planning. Please try again."
        ) from exc

    trace = [TraceStep.model_validate(event) for event in state.get("trace", [])]
    logger.info(
        "plan complete",
        extra={
            "steps": state.get("step_count", 0),
            "finish_reason": state.get("finish_reason"),
            "nodes": [step.node for step in trace],
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "history_turns": len(messages) - 1,
        },
    )
    return PlanResponse(
        request_id=request_id_var.get() or "",
        response=state["messages"][-1].text,
        steps=state.get("step_count", 0),
        finish_reason=state["finish_reason"],
        trace=trace,
    )
