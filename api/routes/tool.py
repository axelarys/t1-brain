from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, Any
import importlib
import logging

tool_router = APIRouter()
logger = logging.getLogger("tool_executor")

class ToolAction(BaseModel):
    action: str
    parameters: Dict[str, Any]

@tool_router.post("/tool/execute")
def execute_tool(action_input: ToolAction):
    from memory.memory_router import MemoryRouter

    action_name = action_input.action
    params = action_input.parameters

    try:
        # 🧩 1. Dynamically load tool module
        module = importlib.import_module(f"tools.{action_name}")
        if not hasattr(module, "run_action"):
            raise Exception("Missing 'run_action' function in tool.")

        # ⚙️ 2. Execute the tool
        result = module.run_action(**params)

        # 🧠 3. Loopback: Store result in memory
        session_id = params.get("session_id", "user_tool_session")
        query = params.get("query", f"{action_name} tool triggered")

        router = MemoryRouter()
        router.memory.store_memory(
            session_id=session_id,
            query=query,
            response=result,
            memory_type="result",
            source_type="tool",
            metadata={ "tool": action_name }
        )

        return { "status": "success", "output": result }

    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
