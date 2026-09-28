import logging
import re
import time
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

# Load backend/.env before any setting is read.
load_dotenv()

from app.config import get_settings  # noqa: E402
from app.graph.workflow import create_workflow, initial_state, run_config  # noqa: E402
from app.logging_config import configure_logging, request_id_var  # noqa: E402
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


# Configure CORS - must be added before routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://192.168.1.199:3000",  # Network access
    ],
    allow_credentials=False,  # Set to False to avoid CORS issues with credentials
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
    max_age=3600,  # Cache preflight responses for 1 hour
)

# Initialize the graph
http_client = create_http_client(get_settings().http_user_agent)
graph = create_workflow(http_client)


class PlanRequest(BaseModel):
    message: str
    history: list[dict] | None = []


@app.options("/plan")
async def plan_options():
    return {"status": "ok"}


@app.post("/plan")
async def plan_trip(request: PlanRequest):
    try:
        max_steps = get_settings().max_steps
        inputs = initial_state([HumanMessage(content=request.message)], max_steps)

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
