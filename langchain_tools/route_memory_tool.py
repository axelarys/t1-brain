import sys
import os
import logging

# Ensure project root is on path for imports
sys.path.append("/root/projects/t1-brain")

from langchain.tools import tool

# 🧠 Logger Setup
LOG_DIR = "/root/projects/t1-brain/logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, "route_memory_tool.log")

route_logger = logging.getLogger("route_memory_logger")
route_logger.setLevel(logging.INFO)

if not route_logger.handlers:
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    route_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    route_logger.addHandler(stream_handler)

# Removed the top-level import of MemoryRouter
# router = MemoryRouter() - this is also moved inside the function

@tool
def route_memory(session_id: str, user_input: str) -> dict:
    """
    Routes a memory query through T1 Brain's internal memory system.

    Args:
        session_id (str): Unique session identifier.
        user_input (str): The query or input to be routed.

    Returns:
        dict: A structured routing response:
        {
            "status": "success" | "error",
            "tool": "route_memory",
            "session_id": <session_id>,
            "query": <user_input>,
            "routing_target": <determined_route>,
            "output": <action_message>
        }
    """
    route_logger.info(f"📨 route_memory called | Session: {session_id} | Input: {user_input}")
    try:
        # Import MemoryRouter inside the function to avoid circular imports
        from memory.memory_router import MemoryRouter
        router = MemoryRouter()
        
        # Enrich and classify the input
        enriched = router.enrich_and_classify(session_id, user_input)
        route_target = enriched.get("storage_target", "unknown")
        route_logger.info(f"🔍 Enrichment result: {enriched}")

        # Fallback simulation if classification is unknown
        if not route_target or route_target == "unknown":
            route_target = "graph" if "relationship" in user_input.lower() else "vector"
            route_logger.info(f"🤖 Fallback classification applied: {route_target}")
            enriched["storage_target"] = route_target

        # Execute the routing action
        result = router.execute_action(
            session_id=session_id,
            user_input=user_input,
            enriched=enriched
        )
        route_logger.info(f"✅ Action executed | Result: {result}")

        return {
            "status": "success",
            "tool": "route_memory",
            "session_id": session_id,
            "query": user_input,
            "routing_target": route_target,
            "output": result.get("message"),
        }

    except Exception as e:
        route_logger.error(f"❌ route_memory failed: {str(e)}")
        return {
            "status": "error",
            "tool": "route_memory",
            "session_id": session_id,
            "query": user_input,
            "error": str(e),
        }