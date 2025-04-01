from fastapi import APIRouter, Request
from pydantic import BaseModel
from memory.session_memory import PersistentSessionMemory
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

# 🧠 Router & Session Memory
router = APIRouter()
memory_handler = PersistentSessionMemory()

# 📦 Standard Memory Input Model
class MemoryRouteRequest(BaseModel):
    session_id: str
    user_input: str

class MemoryStoreRequest(BaseModel):
    session_id: str
    query: str
    response: str
    sentiment: str = "neutral"
    memory_type: str = "semantic"  # default

class MemoryImageStoreRequest(BaseModel):
    session_id: str
    image_url: str
    response: str
    sentiment: str = "neutral"

# 🔁 Route: Standard Text or Semantic Memory
@router.post("/memory/store")
async def store_memory(request: MemoryStoreRequest):
    memory_logger.info(f"📥 /memory/store | session: {request.session_id}")
    result = memory_handler.store_memory(
        session_id=request.session_id,
        query=request.query,
        response=request.response,
        memory_type=request.memory_type,
        sentiment=request.sentiment
    )
    return result

# 🔁 Route: Image Memory (visual understanding)
@router.post("/memory/store_image")
async def store_image_memory(request: MemoryImageStoreRequest):
    memory_logger.info(f"📥 /memory/store_image | session: {request.session_id}")
    result = memory_handler.store_memory(
        session_id=request.session_id,
        query=request.image_url,
        response=request.response,
        memory_type="image",
        sentiment=request.sentiment
    )
    return result
