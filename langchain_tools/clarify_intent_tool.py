import sys
import os
import logging
from langchain.tools import tool
from openai import OpenAI

# 🔧 Path to access settings.py
sys.path.append("/root/projects/t1-brain")
from utils.memory_utils import get_api_key

# 🧠 Logger Setup
LOG_DIR = "/root/projects/t1-brain/logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, "clarify_intent_tool.log")

clarify_logger = logging.getLogger("clarify_intent_logger")
clarify_logger.setLevel(logging.INFO)

if not clarify_logger.handlers:
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    clarify_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    clarify_logger.addHandler(stream_handler)

# 🤖 OpenAI Client with API key based on type
client = OpenAI(api_key=get_api_key("text"))

@tool
def clarify_intent(user_input: str) -> str:
    """
    Used when the user's intent is unclear. Asks a smart, reflective question to understand the memory goal.
    """
    clarify_logger.info(f"📩 Clarify Intent Tool Called | Input: {user_input}")

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

        content = response.choices[0].message.content.strip()
        clarify_logger.info(f"✅ Clarification Response: {content}")
        return content

    except Exception as e:
        clarify_logger.error(f"❌ Clarification failed: {str(e)}")
        return f"❌ Clarification failed: {str(e)}"
