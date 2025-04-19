import sys
import os
import logging
import openai
from langchain.tools import tool

# 🔧 Path to access settings.py
sys.path.append("/root/projects/t1-brain")
from utils.memory_utils import get_api_key

# ✅ Logger Setup (flush-safe, debug-enabled)
clarify_logger = logging.getLogger("clarify_intent_logger")
clarify_logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("/root/projects/t1-brain/logs/clarify_intent_tool.log", mode='a')
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
file_handler.setLevel(logging.DEBUG)

clarify_logger.handlers.clear()
clarify_logger.addHandler(file_handler)
clarify_logger.propagate = False

# ✅ Initialize OpenAI API key
openai.api_key = get_api_key("text")

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

        clarify_logger.info(f"🧠 Prompt Sent to OpenAI: {messages}")

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0
        )

        clarify_logger.info(f"📥 Raw OpenAI Response: {response}")

        try:
            content = response.choices[0].message.content.strip()
            if not content:
                raise ValueError("OpenAI returned an empty string")
        except Exception as parse_error:
            clarify_logger.error(f"❌ Could not parse or received empty OpenAI response: {parse_error}")
            clarify_logger.error(f"📥 Raw OpenAI Response: {response}")
            content = "I didn’t quite understand. Could you clarify your request?"

        clarify_logger.info(f"✅ Clarification Response: {content}")
        return content

    except Exception as e:
        clarify_logger.error(f"❌ Clarification failed: {str(e)}")
        return "I’m not sure what you meant. Could you please clarify your request?"
