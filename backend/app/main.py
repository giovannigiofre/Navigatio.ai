from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from app.config import get_settings
from app.graph.workflow import create_workflow, initial_state, run_config

app = FastAPI(title="Navigatio.ai Agent Backend")

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
graph = create_workflow()

class PlanRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, Any]]] = []

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
        
        return {
            "response": last_message,
            "trace": [m.content for m in messages]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}
