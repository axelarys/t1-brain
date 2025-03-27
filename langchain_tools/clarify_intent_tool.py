import sys
import os
from langchain.tools import tool
from openai import OpenAI

# 🔧 Path to access settings.py
sys.path.append("/root/projects/t1-brain")
from config.settings import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

@tool
def clarify_intent(user_input: str) -> str:
    """
    Used when the user's intent is unclear. Asks a smart, reflective question to understand the memory goal.
    """
    try:
        system_prompt = (
            "You're a helpful memory system. When a query is ambiguous, ask a clarifying question to understand whether "
            "the user is trying to store a memory, retrieve something, update a thought, or delete it. "
            "Avoid rejecting — guide them gently to reveal intent or emotion."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Clarification failed: {str(e)}"
