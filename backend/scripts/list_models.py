"""List the Gemini models available to the configured API key.

Use this to check that GEMINI_MODEL points at a model your key can call.

    python scripts/list_models.py
"""

import os
import sys

from dotenv import load_dotenv
from google import genai


def main() -> int:
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY is not set (see backend/.env.example).", file=sys.stderr)
        return 1

    client = genai.Client(api_key=api_key)
    for model in client.models.list():
        if "generateContent" in (model.supported_actions or []):
            print(model.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
