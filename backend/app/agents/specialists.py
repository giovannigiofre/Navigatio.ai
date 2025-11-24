from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from app.graph.state import TripState

# In a real implementation, these would use tools. 
# For now, we mock them as LLMs with specific system prompts.

def create_agent(llm: ChatGoogleGenerativeAI, system_prompt: str, name: str):
    """Helper to create an agent node."""
    def agent_node(state: TripState):
        messages = state["messages"]
        # We add the system prompt to the history for this specific call
        # In a real graph, we might manage this differently to avoid context bloat
        response = llm.invoke(
            [{"role": "system", "content": system_prompt}] + messages
        )
        return {
            "messages": [AIMessage(content=response.content, name=name)]
        }
    return agent_node

# Specialist Prompts
FLIGHT_PROMPT = """You are an expert Flight Specialist.
Your job is to find the best flight options based on the user's destination and dates.
For this prototype, simulate finding flights. Be specific about airlines, times, and prices.
"""

HOTEL_PROMPT = """You are a Hotel Scout.
Your job is to find accommodation that matches the user's style and budget.
For this prototype, simulate finding hotels. Mention specific hotel names and why they fit.
"""

ACTIVITY_PROMPT = """You are an Activity Curator.
Your job is to find cool things to do.
For this prototype, suggest a mix of tourist highlights and hidden gems.
"""

def get_specialists(llm: ChatGoogleGenerativeAI):
    return {
        "FlightExpert": create_agent(llm, FLIGHT_PROMPT, "FlightExpert"),
        "HotelScout": create_agent(llm, HOTEL_PROMPT, "HotelScout"),
        "ActivityCurator": create_agent(llm, ACTIVITY_PROMPT, "ActivityCurator"),
    }
