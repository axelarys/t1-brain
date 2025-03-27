from fastapi import APIRouter
from pydantic import BaseModel
from langchain_tools.memory_agent import agent
import logging
import asyncio
import os

# 🔁 Router
router = APIRouter()

# 📝 Named Logger (bypasses global logging conflict)
logger = logging.getLogger("agent_logger")
logger.setLevel(logging.INFO)

log_dir = "/root/projects/t1-brain/logs"
log_file = os.path.join(log_dir, "agent.log")
os.makedirs(log_dir, exist_ok=True)

# Avoid duplicate handlers on reloads
if not logger.handlers:
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

# 📦 Request Body
class AgentInput(BaseModel):
    session_id: str
    user_input: str

# 🚀 Endpoint
@router.post("/agent/run")
async def run_agent(request: AgentInput):
    logger.info(f"🟢 Received input: session_id={request.session_id}, query={request.user_input}")

    try:
        result = await asyncio.wait_for(
            agent.ainvoke(f"[{request.session_id}] {request.user_input}"),
            timeout=15
        )

        response_data = {
            "status": "success",
            "response": str(result)
        }

        logger.info(f"✅ Agent responded: {response_data}")
        return response_data

    except asyncio.TimeoutError:
        logger.error("❌ Timeout: Agent did not respond in time.")
        return {
            "status": "error",
            "response": "The memory agent timed out. Please try again."
        }

    except Exception as e:
        logger.error(f"❌ Agent Error: {str(e)}")
        return {
            "status": "error",
            "response": f"Agent failed with: {str(e)}"
        }
