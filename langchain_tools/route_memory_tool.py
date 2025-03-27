import sys
sys.path.append("/root/projects/t1-brain")

from langchain.tools import tool
from memory.memory_router import MemoryRouter

router = MemoryRouter()

@tool
def route_memory(session_id: str, user_input: str) -> str:
    """
    Routes a memory query directly through T1 Brain's internal memory system (no HTTP call).
    Returns: status + route + message string.
    """
    try:
        # 🔍 Enrich + classify
        enriched = router.enrich_and_classify(session_id, user_input)
        route = enriched.get("storage_target", "unknown")

        # 🚀 Perform action (store/retrieve/etc.)
        result = router.execute_action(
            session_id=session_id,
            user_input=user_input,
            enriched=enriched
        )

        return (
            f"✅ {result.get('message')} "
            f"(Route: {result.get('route')})"
        )

    except Exception as e:
        return f"❌ route_memory failed: {str(e)}"
