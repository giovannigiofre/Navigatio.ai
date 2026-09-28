from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List

from app.graph.state import TripState

# Define the system prompt for the supervisor
SYSTEM_PROMPT = """You are the Supervisor of an elite travel concierge team.
Your goal is to orchestrate the creation of a perfect trip itinerary for the user.

You have the following specialists at your disposal:
1. FlightExpert: Finds the best flights.
2. HotelScout: Finds accommodation that matches the user's vibe.
3. ActivityCurator: Finds things to do and places to eat.

Your responsibilities:
1. Analyze the user's request.
2. Delegate tasks to the appropriate specialist.
3. If the plan is complete and satisfies the user's request, respond with "FINISH".
4. If you need more information from the user, respond with "FINISH" and ask the question in the final message.

Do not do the work yourself. Delegate to the specialists.
"""

options = ["FlightExpert", "HotelScout", "ActivityCurator"]

# Define the tool for routing
route_tool = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "type": "object",
        "properties": {
            "next": {
                "type": "string",
                "enum": options,
            }
        },
        "required": ["next"],
    },
}

def create_supervisor_node(llm: ChatGoogleGenerativeAI):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(options), team_members=", ".join(options))

    return (
        prompt
        | llm.bind_tools(tools=[route_tool], tool_choice="route")
        | JsonOutputToolsParser()
        | (lambda x: {"next_agent": x[0]["args"]["next"] if x else "FINISH"})
    )
