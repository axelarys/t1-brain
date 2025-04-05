# agents/tool_agent.py

import importlib
import logging
from typing import Dict, Any, List
from openai import OpenAI
from config import settings

logger = logging.getLogger("tool_agent")

class ToolAgent:
    def __init__(self, session_id: str = "default_session"):
        self.session_id = session_id

        # Lazy memory import to avoid circular import
        from memory.session_memory import PersistentSessionMemory
        self.memory = PersistentSessionMemory()

        # GPT client
        self.client = OpenAI(api_key=settings.LANGCHAIN_OPENAI_API_KEY)

    def run(self, action_schema: Dict[str, Any]) -> Dict[str, Any]:
        action = action_schema.get("action", "")
        parameters = action_schema.get("parameters", {})

        if not action:
            return { "status": "error", "message": "No action provided" }

        if action == "multi_tool_chain":
            return self.run_chain(parameters.get("chain", []))

        if action == "none":
            return {
                "status": "fallback",
                "tool": "none",
                "output": self._run_failsafe_response(parameters)
            }

        return self.run_single_tool(action, parameters)

    def run_single_tool(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        try:
            parameters.setdefault("session_id", self.session_id)
            parameters.setdefault("query", f"Executed action: {action}")

            # 🧠 Inject recent memory context (last 3 Q/As)
            if "memory_context" not in parameters:
                context_snippet = self._get_memory_context(parameters["session_id"])
                if context_snippet:
                    parameters["memory_context"] = context_snippet

            module_path = f"tools.{action}"
            module = importlib.import_module(module_path)

            if not hasattr(module, "run_action"):
                raise ImportError(f"Tool '{action}' does not define a run_action() function")

            result = module.run_action(**parameters)

            return {
                "status": "success",
                "tool": action,
                "input": parameters,
                "output": result
            }

        except Exception as e:
            logger.exception(f"[ToolAgent] Tool '{action}' execution failed")
            return {
                "status": "error",
                "tool": action,
                "input": parameters,
                "message": str(e)
            }

    def run_chain(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not steps or not isinstance(steps, list):
            return { "status": "error", "message": "Invalid tool chain input" }

        history = []
        results = {}

        for i, step in enumerate(steps):
            action = step.get("action")
            params = step.get("parameters", {})

            # 🔁 Replace <result_from_0> placeholders with actual output
            for key, value in params.items():
                if isinstance(value, str) and value.startswith("<result_from_"):
                    try:
                        index = int(value.replace("<result_from_", "").replace(">", ""))
                        params[key] = results.get(index, "")
                    except Exception as e:
                        logger.warning(f"[ToolChain] Failed to resolve placeholder: {e}")
                        params[key] = ""

            result = self.run_single_tool(action, params)
            results[i] = result.get("output", "")
            history.append(result)

        return {
            "status": "chained",
            "steps": len(history),
            "chain": history,
            "final_output": results.get(len(steps) - 1, "")
        }

    def _get_memory_context(self, session_id: str) -> str:
        """Retrieve recent memory snippets for enrichment"""
        try:
            memories = self.memory.retrieve_memory(session_id, query="")
            if not memories:
                return ""
            context_lines = []
            for m in memories[-3:]:
                q, a = m.get("query", ""), m.get("response", "")
                context_lines.append(f"User: {q}\nAssistant: {a}")
            return "\n---\n".join(context_lines)
        except Exception as e:
            logger.warning(f"[ToolAgent] Memory context retrieval failed: {e}")
            return ""

    def _run_failsafe_response(self, parameters: Dict[str, Any]) -> str:
        """Return GPT-generated fallback response as plain string"""
        try:
            query = parameters.get("query", "")
            memory_context = parameters.get("memory_context", "")

            prompt = f"""
You are a helpful assistant. The user's input was not actionable with any predefined tools.

Respond naturally and helpfully to the following:
Context: {memory_context}
User said: "{query}"
"""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    { "role": "system", "content": "Be concise, warm, and conversational. Avoid saying you can't help." },
                    { "role": "user", "content": prompt }
                ],
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"[ToolAgent] Failsafe fallback failed: {e}")
            return f"⚠️ Sorry, I couldn't generate a helpful fallback: {str(e)}"
