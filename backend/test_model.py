from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    res = llm.invoke("Hello")
    print(f"Success with gemini-1.5-pro: {res.content}")
except Exception as e:
    print(f"Error with gemini-1.5-pro: {e}")

try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    res = llm.invoke("Hello")
    print(f"Success with gemini-2.5-pro: {res.content}")
except Exception as e:
    print(f"Error with gemini-2.5-pro: {e}")
