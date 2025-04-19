# main.py

import sys
sys.path.append("/root/projects/t1-brain")
import os
import logging
import redis
import asyncio
from fastapi import FastAPI, Depends, HTTPException, Request
from api.models import ToolMemoryRequest

# Project path
sys.path.append("/root/projects/t1-brain")

# Logging
LOG_DIR = "/root/projects/t1-brain/logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "session_memory.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI()

# HTTP middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logging.info(f"Request received: {request.method} {request.url.path}")
    logging.info(f"Client: {request.client.host if request.client else 'Unknown'}")
    response = await call_next(request)
    logging.info(f"Response status: {response.status_code}")
    return response

# Redis
try:
    redis_client = redis.StrictRedis(host="localhost", port=6379, decode_responses=True)
    redis_client.ping()
    logging.info("✅ Redis connection established.")
except Exception as e:
    logging.error(f"❌ Redis connection failed: {e}")
    raise

# API‑KEY guard
def verify_api_key(request: Request):
    api_key = request.headers.get("X-API-KEY")
    if not api_key or api_key != "WsRocks1234":
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return api_key

# Health
@app.get("/health")
async def health_check():
    return {"status":"healthy","redis":"connected" if redis_client.ping() else "disconnected"}

@app.get("/memory/health")
async def memory_health():
    from memory.warm_layer import WarmMemoryCache
    warm = WarmMemoryCache.get_instance()
    return {
        "status":"warm cache active",
        "vectors_stored": warm.index.ntotal,
        "metadata_entries": len(warm.metadata)
    }

@app.get("/debug/routes")
async def debug_routes():
    return {"routes":[{"path":r.path,"methods":list(r.methods)} for r in app.routes]}

# Tool memory endpoint
@app.post("/tool/memory")
async def exec_tool_memory(
    request: ToolMemoryRequest,
    api_key: str = Depends(verify_api_key)
):
    logging.info(f"🔧 /tool/memory | session={request.session_id} | input={request.user_input}")
    try:
        from memory.memory_router import MemoryRouter
        router = MemoryRouter()
        result = router.route_user_query(request.session_id, request.user_input)
        return result
    except Exception as e:
        logging.exception("❌ /tool/memory internal error")
        return {"status": "error", "message": str(e)}

# Mount your other routers
from api.routes import memory, agent
from api.routes.tool import tool_router
from api.routes.health import health_router  # ✅ New

app.include_router(memory.router, prefix="")
app.include_router(agent.router, prefix="")
app.include_router(tool_router, prefix="")
app.include_router(health_router, prefix="")  # ✅ New

# Periodic FAISS saver
async def save_warm_cache_periodically(interval=300):
    from memory.warm_layer import WarmMemoryCache
    warm_cache = WarmMemoryCache.get_instance()
    while True:
        await asyncio.sleep(interval)
        warm_cache.save_index()
        logging.info("[FAISS] ⏳ Periodic warm cache save completed.")

@app.on_event("startup")
async def startup_event():
    logging.info("🚀 FastAPI server started")
    from memory.warm_layer import WarmMemoryCache
    warm_cache = WarmMemoryCache.get_instance()
    logging.info(f"[FAISS] Warm cache primed with {warm_cache.index.ntotal} vectors")
    asyncio.create_task(save_warm_cache_periodically())
    memory.init_dependencies()
    agent.init_memory_handler()

@app.on_event("shutdown")
async def shutdown_event():
    logging.info("🛑 Shutting down")
    from memory.warm_layer import WarmMemoryCache
    WarmMemoryCache.get_instance().save_index()
    logging.info("[FAISS] 💾 Warm memory index saved on shutdown")
