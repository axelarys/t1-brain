# actions/schema_parser.py

import openai
import logging
import json
import re
from typing import Dict, Any
import config.settings as settings  # Use constants from config/settings.py

# Set your OpenAI API key from config constant
openai.api_key = settings.LANGCHAIN_OPENAI_API_KEY

# Escaped-brace version of the hybrid prompt template
PROMPT_TEMPLATE = """
You are an intelligent assistant. Convert user queries into a structured action schema.

Return ONLY valid JSON:
{{
  "action": "string",
  "parameters": {{ ... }}
}}

If no actionable intent is detected, return:
{{ "action": "none", "parameters": {{}} }}

Use available tools: {tools}
Use this memory/context if needed:
"{memory_snippet}"

Now parse this:
"{user_input}"
"""

def clean_json_string(raw: str) -> str:
    """Remove markdown-style ```json wrappers or code block fences."""
    return re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.IGNORECASE).strip()

def generate_action_schema(user_input: str, memory_snippet: str = "", tools: list[str] = None) -> Dict[str, Any]:
    if tools is None:
        tools = ["set_reminder", "summarize_file", "web_search"]

    prompt = PROMPT_TEMPLATE.format(
        tools=tools,
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
        raw = call_gpt(prompt)
        print("\n🪵 Raw GPT Output (1st try):\n", raw)
        return json.loads(clean_json_string(raw))

    except Exception as e1:
        logging.warning(f"[GPT Action Parser] First attempt failed: {e1}")
        retry_prompt = prompt + "\n\nIMPORTANT: Return ONLY compact JSON and nothing else."

        try:
            raw_retry = call_gpt(retry_prompt)
            print("\n🪵 Raw GPT Output (Retry):\n", raw_retry)
            return json.loads(clean_json_string(raw_retry))

        except Exception as e2:
            logging.warning(f"[GPT Action Parser] Retry failed: {e2}")
            try:
                # Final fallback using eval (only in dev)
                return eval(clean_json_string(raw_retry))
            except Exception as e3:
                logging.error(f"[GPT Action Parser] All parsing failed: {e3}")
                return {
                    "action": "none",
                    "parameters": {},
                    "error": str(e3)
                }
