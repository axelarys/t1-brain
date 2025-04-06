# actions/schema_parser.py

import openai
import logging
import json
from typing import Dict, Any
import config.settings as settings

from utils.tool_discovery import discover_tools  # ✅ Dynamic discovery

# Set your OpenAI API key
openai.api_key = settings.LANGCHAIN_OPENAI_API_KEY

# Prompt template with escaped braces
PROMPT_TEMPLATE = """
You are an intelligent assistant. Convert user queries into a structured action schema.

Return ONLY valid JSON:
{{
  "action": "string",
  "parameters": {{ ... }}
}}

If no actionable intent is detected, return:
{{ "action": "none", "parameters": {{}} }}

Use available tools:
{tools}

Use this memory/context if needed:
"{memory_snippet}"

Now parse this:
"{user_input}"
"""

def format_tool_list(tools: list[dict]) -> str:
    """Compact tool format for GPT prompt"""
    return "\n".join([f"- {t['name']}: {t['description'].splitlines()[0]}" for t in tools])

def generate_action_schema(user_input: str, memory_snippet: str = "", fallback_tools: list[str] = None) -> Dict[str, Any]:
    try:
        tool_metadata = discover_tools()
        tools_prompt = format_tool_list(tool_metadata)
    except Exception:
        # Fallback tools in case discovery fails
        fallback_tools = fallback_tools or ["set_reminder", "file_reader", "web_search"]
        tools_prompt = ", ".join(fallback_tools)

    prompt = PROMPT_TEMPLATE.format(
        tools=tools_prompt,
        memory_snippet=memory_snippet,
        user_input=user_input
    )

    def call_gpt(prompt_text: str) -> str:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                { "role": "system", "content": "You are an intelligent action parser." },
                { "role": "user", "content": prompt_text }
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    try:
        output = call_gpt(prompt)
        return json.loads(output)
    except Exception as e1:
        logging.warning(f"[GPT Action Parser] First attempt failed: {e1}")
        try:
            retry_prompt = prompt + "\n\nIMPORTANT: Return ONLY compact JSON and nothing else."
            output_retry = call_gpt(retry_prompt)
            return json.loads(output_retry)
        except Exception as e2:
            logging.error(f"[GPT Action Parser] Retry also failed: {e2}")
            return {
                "action": "none",
                "parameters": {},
                "error": str(e2)
            }
