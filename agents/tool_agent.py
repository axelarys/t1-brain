import logging
import time
import inspect
from typing import Any, Callable, Dict

logger = logging.getLogger("tool_agent")


class ToolAgent:
    """
    Executes available tools by name using the provided schema.
    Supports automatic retry, input logging, output logging, and schema discovery.
    """

    def __init__(self):
        self.registry: Dict[str, Callable[[dict], dict]] = {}

        try:
            from langchain_tools.route_memory_tool import route_memory
            self.registry["route_memory"] = route_memory
        except ImportError as e:
            logger.warning(f"[ToolAgent] ⚠️ Failed to load 'route_memory': {e}")

        try:
            from langchain_tools.clarify_intent_tool import clarify_intent
            self.registry["clarify_intent"] = clarify_intent
        except ImportError as e:
            logger.warning(f"[ToolAgent] ⚠️ Failed to load 'clarify_intent': {e}")

        try:
            from tools.set_reminder import set_reminder
            self.registry["set_reminder"] = set_reminder
        except ImportError as e:
            logger.warning(f"[ToolAgent] ⚠️ Failed to load 'set_reminder': {e}")

        try:
            from tools.file_reader import file_reader
            self.registry["file_reader"] = file_reader
        except ImportError as e:
            logger.warning(f"[ToolAgent] ⚠️ Failed to load 'file_reader': {e}")

        logger.info(f"[ToolAgent] 🛠️ Registered tools: {list(self.registry.keys())}")

    def discover_schema(self, tool_name: str) -> dict:
        """
        Attempts to extract input schema or docstring from the tool.
        """
        tool = self.registry.get(tool_name)
        if not tool:
            return {"status": "error", "message": f"Tool '{tool_name}' not found"}

        try:
            doc = inspect.getdoc(tool)
            sig = inspect.signature(tool)
            params = {
                k: str(v.annotation) if v.annotation != inspect._empty else "Any"
                for k, v in sig.parameters.items()
                if k != "self"
            }

            return {
                "status": "success",
                "tool": tool_name,
                "doc": doc or "No docstring available.",
                "input_schema": params,
            }
        except Exception as e:
            logger.warning(f"[ToolAgent] Schema discovery failed for {tool_name}: {e}")
            return {"status": "error", "message": str(e)}

    def run(self, schema: dict, retry: int = 1) -> dict:
        """
        Runs a registered tool with retry logic and latency tracking.

        Args:
            schema (dict): Must contain "tool" and additional required fields.
            retry (int): Number of retry attempts on failure.

        Returns:
            dict: Execution result with latency and retry info.
        """
        tool_name = schema.get("tool")
        if not tool_name:
            return {"status": "error", "message": "Missing tool name in schema"}

        if tool_name not in self.registry:
            return {
                "status": "error",
                "message": f"Tool '{tool_name}' not registered",
            }

        tool = self.registry[tool_name]

        for attempt in range(1, retry + 1):
            try:
                start = time.time()
                logger.info(f"[ToolAgent] 🚀 Running '{tool_name}' (Attempt {attempt})")
                result = tool(schema)
                end = time.time()

                latency_ms = int((end - start) * 1000)
                logger.info(f"[ToolAgent] ✅ Tool '{tool_name}' completed in {latency_ms}ms")

                if isinstance(result, str):
                    result = {"output": result}

                result.update({
                    "status": "success",
                    "tool": tool_name,
                    "input": schema,
                    "latency_ms": latency_ms,
                    "retry_attempt": attempt,
                })
                return result

            except Exception as e:
                logger.warning(f"[ToolAgent] ⚠️ Tool '{tool_name}' failed on attempt {attempt}: {e}")
                if attempt == retry:
                    return {
                        "status": "error",
                        "tool": tool_name,
                        "message": str(e),
                        "retry_attempt": attempt,
                    }


def get_tool_agent() -> ToolAgent:
    return ToolAgent()