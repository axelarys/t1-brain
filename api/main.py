import sys
import os
import logging
import redis
from fastapi import FastAPI, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import Optional

# ✅ Corrected path for settings.py
sys.path.append("/root/projects/t1-brain/config")
from settings import PG_HOST, PG_DATABASE, PG_USER, PG_PASSWORD

# ✅ Corrected path for session_memory
sys.path.append("/root/projects/t1-brain/memory")
from session_memory import PersistentSessionMemory

# ✅ Logging setup with fallback
LOG_DIR = "/root/projects/t1-brain/logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=f"{LOG_DIR}/session_memory.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ✅ Initialize FastAPI
app = FastAPI()

# ✅ Initialize memory handler
memory_handler = PersistentSessionMemory()

# ✅ Redis connection
try:
    redis_client = redis.StrictRedis(host="localhost", port=6379, decode_responses=True)
    redis_client.ping()
    logging.info("✅ Redis connection established.")
except Exception as e:
    logging.error(f"❌ Redis connection failed: {str(e)}")
    raise

# ✅ API Key dependency
def verify_api_key(request: Request):
    api_key = request.headers.get("X-API-KEY")
    if not api_key:
        raise HTTPException(status_code=401, detail="API Key missing")
    return api_key

# ✅ Pydantic models
class MemoryRequest(BaseModel):
    session_id: str
    query: str
    response: Optional[str] = None
    memory_type: Optional[str] = "semantic"
    sentiment: Optional[str] = "neutral"

class MemoryDeleteRequest(BaseModel):
    session_id: str
    query: str

# ✅ Endpoints
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

# ✅ Entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
