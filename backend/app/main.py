from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from app.graph.workflow import create_workflow

app = FastAPI(title="Navigatio.ai Agent Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the graph
graph = create_workflow()

class PlanRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, Any]]] = []

@app.post("/plan")
async def plan_trip(request: PlanRequest):
    try:
        # Convert input to LangChain messages
        inputs = {
            "messages": [HumanMessage(content=request.message)]
        }
        
        # Run the graph
        # In a real app, we would stream this or use a background task
        final_state = await graph.ainvoke(inputs)
        
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
