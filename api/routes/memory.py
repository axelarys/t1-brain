from fastapi import APIRouter, Request
from pydantic import BaseModel
from memory.memory_router import MemoryRouter
import logging
import os

# 🧠 Logger Setup
LOG_DIR = "/root/projects/t1-brain/logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, "memory_router_http.log")

memory_logger = logging.getLogger("memory_router_logger")
memory_logger.setLevel(logging.INFO)

if not memory_logger.handlers:
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    memory_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    memory_logger.addHandler(stream_handler)

# 🧩 Router & Model
router = APIRouter()
memory_router = MemoryRouter()

class MemoryRouteRequest(BaseModel):
    session_id: str
    user_input: str

@router.post("/memory/route")
async def route_and_execute_memory(request: MemoryRouteRequest):
    memory_logger.info(f"📥 Received route request | Session: {request.session_id} | Input: {request.user_input}")

    try:
        classification = memory_router.enrich_and_classify(
            user_id=request.session_id, user_input=request.user_input
        )
        memory_logger.info(f"🔎 Classified: {classification}")

        result = memory_router.execute_action(
            session_id=request.session_id,
            user_input=request.user_input,
            enriched=classification
        )
        memory_logger.info(f"✅ Action result: {result}")

        return {
            "status": result.get("status"),
            "route": result.get("route"),
            "message": result.get("message"),
            "matches": result.get("matches", []),
            "meta": classification
        }

    except Exception as e:
        memory_logger.error(f"❌ Memory routing failed: {str(e)}")
        return {"status": "error", "message": str(e)}
