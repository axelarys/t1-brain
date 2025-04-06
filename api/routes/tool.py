# api/routes/tool.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import importlib
import sys
import logging

from memory.memory_router import MemoryRouter
from agents.tool_agent import ToolAgent
from utils.tool_discovery import discover_tools  # ✅ Added

tool_router = APIRouter()
logger = logging.getLogger("tool_executor")

class ToolAction(BaseModel):
    action: str
    parameters: Dict[str, Any]

@tool_router.post("/tool/execute")
def execute_tool(action_input: ToolAction):
    action_name = action_input.action
    params = action_input.parameters

    # ⚠️ Fallback: Non-actionable input
    if action_name == "none":
        logger.warning("[ToolExecutor] No actionable tool detected — fallback engaged.")
        agent = ToolAgent(session_id=params.get("session_id", "failsafe"))
        fallback = agent._run_failsafe_response(params)
        return fallback

    try:
        # 🔁 Reload tool module if already cached
        module_path = f"tools.{action_name}"
        if module_path in sys.modules:
            del sys.modules[module_path]
        module = importlib.import_module(module_path)

        if not hasattr(module, "run_action"):
            raise Exception("Missing 'run_action' function in tool.")

        result = module.run_action(**params)

        # 🧠 Store in memory
        session_id = params.get("session_id", "user_tool_session")
        query = params.get("query", f"{action_name} tool triggered")

        router = MemoryRouter()
        router.memory.store_memory(
            session_id=session_id,
            query=query,
            response=result,
            memory_type="result",
            source_type="tool",
            metadata={"tool": action_name}
        )

        return { "status": "success", "output": result }

    except Exception as e:
        logger.error(f"[ToolExecutor] Execution failed for '{action_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))

@tool_router.get("/tool/list")
def list_available_tools():
    try:
        tools = discover_tools()
        return { "status": "success", "tools": tools }
    except Exception as e:
        logger.error(f"[ToolList] Failed to list tools: {e}")
        raise HTTPException(status_code=500, detail="Failed to list tools")
