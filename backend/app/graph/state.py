from typing import TypedDict, Annotated, List, Optional, Dict, Any
import operator
from langchain_core.messages import BaseMessage

class TripState(TypedDict):
    """
    The global state of the trip planning process.
    """
    messages: Annotated[List[BaseMessage], operator.add]
    destination: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    budget: Optional[str]
    
    # The current working itinerary draft
    itinerary: Optional[Dict[str, Any]]
    
    # Track which agent is currently acting or next to act
    next_agent: Optional[str]
    
    # Error tracking
    error: Optional[str]
