"""Manual smoke run of the real graph against Gemini (needs GOOGLE_API_KEY).

cd backend && python -m scripts.smoke_graph "Weekend in Lisbon next month, flexible budget"
"""

import asyncio
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from app.config import get_settings
from app.graph.workflow import create_workflow, initial_state, run_config
from app.tools.http import create_http_client

DEFAULT_REQUEST = (
    "I want to go to Paris for a weekend in May. I like art and good food. Budget is flexible."
)


async def main(user_input: str) -> None:
    load_dotenv()
    settings = get_settings()
    state = initial_state([HumanMessage(content=user_input)], settings.max_steps)

    print(f"User: {user_input}\n" + "-" * 50)
    async with create_http_client(settings.http_user_agent) as http:
        graph = create_workflow(http, settings)
        async for update in graph.astream(
            state, config=run_config(settings.max_steps), stream_mode="updates"
        ):
            for node, value in update.items():
                print(f"[{node}] {value}\n" + "-" * 20)


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or DEFAULT_REQUEST))
