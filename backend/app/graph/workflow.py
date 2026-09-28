from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import get_settings
from app.graph.state import TripState
from app.agents.supervisor import create_supervisor_node
from app.agents.specialists import get_specialists

def create_workflow():
    # Initialize LLM
    # Note: Requires GOOGLE_API_KEY in environment
    llm = ChatGoogleGenerativeAI(model=get_settings().gemini_model)
    
    # Create Nodes
    supervisor_node = create_supervisor_node(llm)
    specialists = get_specialists(llm)
    
    # Define Graph
    workflow = StateGraph(TripState)
    
    workflow.add_node("Supervisor", supervisor_node)
    workflow.add_node("FlightExpert", specialists["FlightExpert"])
    workflow.add_node("HotelScout", specialists["HotelScout"])
    workflow.add_node("ActivityCurator", specialists["ActivityCurator"])
    
    # Define Edges
    for member in specialists:
        # After a specialist finishes, go back to supervisor
        workflow.add_edge(member, "Supervisor")
    
    # The supervisor decides where to go next
    conditional_map = {k: k for k in specialists}
    conditional_map["FINISH"] = END
    
    workflow.add_conditional_edges(
        "Supervisor",
        lambda x: x["next"] if x["next"] in conditional_map else "FINISH",
        conditional_map
    )
    
    workflow.set_entry_point("Supervisor")
    
    return workflow.compile()
