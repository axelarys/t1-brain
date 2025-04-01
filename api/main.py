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
from memory.memory_router import MemoryRouter  # 🔁 Direct memory routing logic

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

# Middleware to log incoming requests
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
    query: str
    response: Optional[str] = None
    memory_type: Optional[str] = "semantic"
    sentiment: Optional[str] = "neutral"

class MemoryDeleteRequest(BaseModel):
    session_id: str
    query: str

class ToolMemoryRequest(BaseModel):  # Used only for internal routing test (optional)
    session_id: str
    user_input: str

# 🧠 Core API Endpoints
@app.post("/memory/store")
async def api_store_memory(request: MemoryRequest, api_key: str = Depends(verify_api_key)):
    logging.info(f"API store memory called: {request.session_id}")
    return memory_handler.store_memory(
        request.session_id, request.query, request.response, request.memory_type, request.sentiment
    )

@app.post("/memory/retrieve")
async def api_retrieve_memory(request: MemoryRequest, api_key: str = Depends(verify_api_key)):
    logging.info(f"API retrieve memory called: {request.session_id}")
    return {
        "status": "retrieved",
        "memory": memory_handler.retrieve_memory(request.session_id, request.query)
    }

@app.delete("/memory/delete")
async def api_delete_memory(request: MemoryDeleteRequest, api_key: str = Depends(verify_api_key)):
    logging.info(f"API delete memory called: {request.session_id}")
    return memory_handler.delete_memory(request.session_id, request.query)

@app.get("/health")
async def health_check():
    logging.info("Health check called")
    return {
        "status": "healthy",
        "redis": "connected" if redis_client.ping() else "disconnected"
    }

# Debug endpoint to list all registered routes
@app.get("/debug/routes")
async def debug_routes():
    logging.info("Debug routes called")
    routes = []
    for route in app.routes:
        routes.append({
            "path": route.path,
            "name": route.name,
            "methods": route.methods
        })
    return {"routes": routes}

# 🧪 Internal Debug Endpoint (can remove for final prod)
@app.post("/tool/memory")
async def test_route_memory_tool(request: ToolMemoryRequest):
    logging.info(f"Tool memory called: {request.session_id}")
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
        logging.exception("❌ route_memory_tool direct call failed")
        return {"status": "error", "message": str(e)}

# 🔗 Register Routes
app.include_router(memory.router, prefix="")  # Ensure no prefix
app.include_router(agent.router, prefix="")   # Ensure no prefix

# Startup event to log when the server starts
@app.on_event("startup")
async def startup_event():
    logging.info("🚀 FastAPI server started")
    # Print all registered routes for debugging
    for route in app.routes:
        logging.info(f"Registered route: {route.path} - {route.methods}")