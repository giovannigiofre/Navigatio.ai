import logging
import re
import time
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

# Load backend/.env before any setting is read.
load_dotenv()

from app.config import get_settings  # noqa: E402
from app.conversation import build_conversation  # noqa: E402
from app.graph.workflow import create_workflow, initial_state, run_config  # noqa: E402
from app.logging_config import configure_logging, request_id_var  # noqa: E402
from app.schemas import PlanRequest  # noqa: E402
from app.tools.http import create_http_client  # noqa: E402

configure_logging(get_settings().log_level)
logger = logging.getLogger("app.api")

app = FastAPI(title="Navigatio.ai Agent Backend")

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

# Initialize the graph
http_client = create_http_client(get_settings().http_user_agent)
graph = create_workflow(http_client)


@app.post("/plan")
async def plan_trip(request: PlanRequest):
    try:
        settings = get_settings()
        max_steps = settings.max_steps
        messages = build_conversation(request.history, request.message, settings.max_history_turns)
        inputs = initial_state(messages, max_steps)

        # Run the graph
        # In a real app, we would stream this or use a background task
        final_state = await graph.ainvoke(inputs, config=run_config(max_steps))

        # Extract the final response
        messages = final_state["messages"]
        last_message = messages[-1].content

        return {"response": last_message, "trace": [m.content for m in messages]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/health")
async def health():
    return {"status": "ok"}
