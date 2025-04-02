from fastapi import APIRouter, Request
from pydantic import BaseModel
import logging, os, asyncio
from typing import Optional, Union
from langchain_tools.memory_agent import agent
from memory.session_memory import PersistentSessionMemory

# 🔁 Router
router = APIRouter()

# 🧠 Memory Handler
memory_handler = PersistentSessionMemory()

# 📝 Logger
logger = logging.getLogger("agent_logger")
logger.setLevel(logging.INFO)

log_dir = "/root/projects/t1-brain/logs"
log_file = os.path.join(log_dir, "agent.log")
os.makedirs(log_dir, exist_ok=True)

if not logger.handlers:
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

# 📦 Request Model for multimodal input
class AgentInput(BaseModel):
    session_id: str
    input: dict  # expects {"text": "..."} or {"image_url": "..."}

# 🚀 GPT Actions Entry Point
@router.post("/agent/run")
async def run_agent(request: AgentInput):
    session_id = request.session_id
    user_input = request.input

    # 🔍 Image Input Flow
    if "image_url" in user_input:
        image_url = user_input["image_url"]
        logger.info(f"🖼️ Image input detected: session={session_id}, image_url={image_url}")

        try:
            result = memory_handler.store_memory(
                session_id=session_id,
                query=image_url,
                response="",  # No user text response
                memory_type="image",
                sentiment="neutral"
            )
            logger.info(f"✅ Image memory stored: {result}")
            return {
                "status": "success",
                "response": "Image processed and stored as memory.",
                "source_type": "image"
            }
        except Exception as e:
            logger.error(f"❌ Image memory store failed: {e}")
            return {
                "status": "error",
                "response": f"Image memory failed: {str(e)}",
                "source_type": "image"
            }