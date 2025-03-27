import sys
import os
import logging
import redis
from fastapi import FastAPI, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import Optional

# 🔧 Force project root into path
sys.path.append("/root/projects/t1-brain")

# ✅ Correct imports from project root
from config.settings import PG_HOST, PG_DATABASE, PG_USER, PG_PASSWORD
from memory.session_memory import PersistentSessionMemory
from api.routes import memory   # /memory/route
from api.routes import agent    # /agent/run

# 📂 Logging setup
LOG_DIR = "/root/projects/t1-brain/logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "session_memory.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 🚀 FastAPI App Init
app = FastAPI()

# 🧠 Memory Handler
memory_handler = PersistentSessionMemory()

# 🔗 Redis Connection
try:
    redis_client = redis.StrictRedis(host="localhost", port=6379, decode_responses=True)
    redis_client.ping()
    logging.info("✅ Redis connection established.")
except Exception as e:
    logging.error(f"❌ Redis connection failed: {str(e)}")
    raise

# 🔐 API Key Dependency
def verify_api_key(request: Request):
    api_key = request.headers.get("X-API-KEY")
    if not api_key or api_key != "WsRocks1234":  # 🔐 Static API key for internal use
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return api_key


# 📦 Pydantic Models
class MemoryRequest(BaseModel):
    session_id: str
    query: str
    response: Optional[str] = None
    memory_type: Optional[str] = "semantic"
    sentiment: Optional[str] = "neutral"

class MemoryDeleteRequest(BaseModel):
    session_id: str
    query: str

# 🧠 Core Endpoints
@app.post("/memory/store")
async def api_store_memory(request: MemoryRequest, api_key: str = Depends(verify_api_key)):
    return memory_handler.store_memory(
        request.session_id, request.query, request.response, request.memory_type, request.sentiment
    )

@app.post("/memory/retrieve")
async def api_retrieve_memory(request: MemoryRequest, api_key: str = Depends(verify_api_key)):
    return {
        "status": "retrieved",
        "memory": memory_handler.retrieve_memory(request.session_id, request.query)
    }

@app.delete("/memory/delete")
async def api_delete_memory(request: MemoryDeleteRequest, api_key: str = Depends(verify_api_key)):
    return memory_handler.delete_memory(request.session_id, request.query)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "redis": "connected" if redis_client.ping() else "disconnected"
    }

# 🧩 Register Routers
app.include_router(memory.router)
app.include_router(agent.router)
