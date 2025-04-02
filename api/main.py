import sys
import os
import logging
import redis
from fastapi import FastAPI, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import Optional

# 🔧 Project Path Setup
sys.path.append("/root/projects/t1-brain")

# ✅ Imports
from config.settings import PG_HOST, PG_DATABASE, PG_USER, PG_PASSWORD
from memory.session_memory import PersistentSessionMemory
from api.routes import memory   # /memory/route
from api.routes import agent    # /agent/run
from memory.memory_router import MemoryRouter

# 📂 Logging Setup
LOG_DIR = "/root/projects/t1-brain/logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "session_memory.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 🚀 FastAPI App
app = FastAPI()

@app.middleware("http")
async def log_requests(request, call_next):
    logging.info(f"Request received: {request.method} {request.url.path}")
    logging.info(f"Client: {request.client.host if request.client else 'Unknown'}")
    response = await call_next(request)
    logging.info(f"Response status: {response.status_code}")
    return response

# 🧠 Memory Layer
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
    if not api_key or api_key != "WsRocks1234":
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return api_key

# 📦 Pydantic Models
class MemoryRequest(BaseModel):
    session_id: str
    query: str  # Can be image_url if memory_type is image
    response: Optional[str] = None
    memory_type: Optional[str] = "semantic"
    sentiment: Optional[str] = "neutral"

class MemoryDeleteRequest(BaseModel):
    session_id: str
    query: str

class ToolMemoryRequest(BaseModel):  # Optional internal test
    session_id: str
    user_input: str

# 🧠 Core API Endpoints
@app.post("/memory/store")
async def api_store_memory(request: MemoryRequest, api_key: str = Depends(verify_api_key)):
    logging.info(f"📥 /memory/store | session={request.session_id} | type={request.memory_type}")
    return memory_handler.store_memory(
        session_id=request.session_id,
        query=request.query,
        response=request.response,
        memory_type=request.memory_type,
        sentiment=request.sentiment
    )

@app.post("/memory/retrieve")
async def api_retrieve_memory(request: MemoryRequest, api_key: str = Depends(verify_api_key)):
    logging.info(f"📤 /memory/retrieve | session={request.session_id}")
    raw_memories = memory_handler.retrieve_memory(request.session_id, request.query)

    # Add image_url and source_type if present
    memory_list = []
    for m in raw_memories:
        parsed = {
            "query": m.get("query"),
            "response": m.get("response"),
            "sentiment": m.get("sentiment"),
            "memory_type": m.get("memory_type"),
            "timestamp": m.get("timestamp"),
        }
        if m.get("source_type") == "image":
            parsed["source_type"] = "image"
            parsed["image_url"] = m.get("image_url")
        memory_list.append(parsed)

    return {
        "status": "retrieved",
        "count": len(memory_list),
        "memory": memory_list
    }

@app.delete("/memory/delete")
async def api_delete_memory(request: MemoryDeleteRequest, api_key: str = Depends(verify_api_key)):
    logging.info(f"❌ /memory/delete | session={request.session_id}")
    return memory_handler.delete_memory(request.session_id, request.query)

@app.get("/health")
async def health_check():
    logging.info("🩺 /health check")
    return {
        "status": "healthy",
        "redis": "connected" if redis_client.ping() else "disconnected"
    }

@app.get("/debug/routes")
async def debug_routes():
    logging.info("🔍 /debug/routes")
    return {
        "routes": [{
            "path": route.path,
            "name": route.name,
            "methods": list(route.methods)
        } for route in app.routes]
    }

@app.post("/tool/memory")
async def test_route_memory_tool(request: ToolMemoryRequest):
    logging.info(f"🔧 /tool/memory | session={request.session_id}")
    try:
        router = MemoryRouter()
        enriched = router.enrich_and_classify(request.session_id, request.user_input)
        result = router.execute_action(
            session_id=request.session_id,
            user_input=request.user_input,
            enriched=enriched
        )
        return {"status": "success", "enriched": enriched, "result": result}
    except Exception as e:
        logging.exception("❌ /tool/memory internal error")
        return {"status": "error", "message": str(e)}

# 🔗 Route Mounting
app.include_router(memory.router, prefix="")
app.include_router(agent.router, prefix="")

@app.on_event("startup")
async def startup_event():
    logging.info("🚀 FastAPI server started")
    for route in app.routes:
        logging.info(f"Registered route: {route.path} - {route.methods}")
