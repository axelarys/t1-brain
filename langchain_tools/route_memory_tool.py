import sys
import os
import logging

sys.path.append("/root/projects/t1-brain")

from langchain.tools import tool
from memory.memory_router import MemoryRouter

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

# 🛠️ Tool Logic
router = MemoryRouter()

@tool
def route_memory(session_id: str, user_input: str) -> str:
    """
    Routes a memory query directly through T1 Brain's internal memory system (no HTTP call).
    Returns: status + route + message string.
    """
    route_logger.info(f"📨 route_memory called | Session: {session_id} | Input: {user_input}")

    try:
        enriched = router.enrich_and_classify(session_id, user_input)
        route = enriched.get("storage_target", "unknown")
        route_logger.info(f"🔍 Classification: {enriched}")

        if route == "unknown":
            route_logger.warning("⚠️ Enrichment failed to determine proper route.")

        result = router.execute_action(
            session_id=session_id,
            user_input=user_input,
            enriched=enriched
        )

        route_logger.info(f"✅ Action executed | Result: {result}")
        return (
            f"✅ {result.get('message')} "
            f"(Route: {result.get('route')})"
        )

    except Exception as e:
        route_logger.error(f"❌ route_memory failed: {str(e)}")
        return f"❌ route_memory failed: {str(e)}"
