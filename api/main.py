import sys
import logging
import redis
import psycopg2
from fastapi import FastAPI, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, Dict

# Ensure Python can locate settings.py
sys.path.append("/root/projects/t1-brain/config")
from settings import PG_HOST, PG_DATABASE, PG_USER, PG_PASSWORD

# Ensure Python can locate session_memory.py
sys.path.append("/root/projects/t1-brain/memory")
from memory.session_memory import PersistentSessionMemory

# Setup Logging (with updated path)
LOG_DIR = "/root/projects/t1-brain/logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=f"{LOG_DIR}/session_memory.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize FastAPI
app = FastAPI()

# Initialize Memory Handler
memory_handler = PersistentSessionMemory()

# Initialize Redis
try:
    redis_client = redis.StrictRedis(host="localhost", port=6379, decode_responses=True)
    logging.info("✅ Redis connection established.")
except Exception as e:
    logging.error(f"❌ Redis connection failed: {str(e)}")
    raise

# API Key Validation
def verify_api_key(request: Request):
    api_key = request.headers.get("X-API-KEY")
    if not api_key:
        raise HTTPException(status_code=401, detail="API Key missing")
    return api_key

# API Endpoints
class MemoryRequest(BaseModel):
    session_id: str
    query: str
    response: Optional[str] = None
    memory_type: Optional[str] = "semantic"
    sentiment: Optional[str] = "neutral"

class MemoryDeleteRequest(BaseModel):
    session_id: str
    query: str

@app.post("/memory/store")
async def api_store_memory(request: MemoryRequest, api_key: str = Depends(verify_api_key)):
    return memory_handler.store_memory(request.session_id, request.query, request.response, request.memory_type, request.sentiment)

@app.post("/memory/retrieve")
async def api_retrieve_memory(request: MemoryRequest, api_key: str = Depends(verify_api_key)):
    return {"status": "retrieved", "memory": memory_handler.retrieve_memory(request.session_id, request.query)}

@app.delete("/memory/delete")
async def api_delete_memory(request: MemoryDeleteRequest, api_key: str = Depends(verify_api_key)):
    return memory_handler.delete_memory(request.session_id, request.query)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "database": "connected" if redis_client.ping() else "disconnected"}

# Run FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
