import asyncio
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from app.graph.workflow import create_workflow

# Load environment variables (for GOOGLE_API_KEY)
load_dotenv()

async def main():
    print("Initializing Graph...")
    try:
        graph = create_workflow()
    except Exception as e:
        print(f"Error initializing graph: {e}")
        return

    print("Graph Initialized. Running test query...")
    
    user_input = "I want to go to Paris for a weekend in May. I like art and good food. Budget is flexible."
    
    inputs = {
        "messages": [HumanMessage(content=user_input)]
    }
    
    print(f"User: {user_input}")
    print("-" * 50)
    
    try:
        async for output in graph.astream(inputs):
            for key, value in output.items():
                print(f"Node '{key}' finished.")
                print(f"Output: {value}")
                print("-" * 20)
    except Exception as e:
        print(f"Error running graph: {e}")

if __name__ == "__main__":
    asyncio.run(main())
