import sys
import os
import logging
import redis
from fastapi import FastAPI, Depends, HTTPException, Request

# 🔧 Project Path Setup
sys.path.append("/root/projects/t1-brain")

# 📂 Logging Setup
LOG_DIR = "/root/projects/t1-brain/logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "session_memory.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Import models first (no circular dependencies here)
from api.models import ToolMemoryRequest

# 🚀 FastAPI App
app = FastAPI()

@app.middleware("http")
async def log_requests(request, call_next):
    logging.info(f"Request received: {request.method} {request.url.path}")
    logging.info(f"Client: {request.client.host if request.client else 'Unknown'}")
    response = await call_next(request)
    logging.info(f"Response status: {response.status_code}")
    return response

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

# 🩺 Health Check
@app.get("/health")
async def health_check():
    logging.info("🩺 /health check")
    return {
        "status": "healthy",
        "redis": "connected" if redis_client.ping() else "disconnected"
    }

# 🔍 Debug Routes
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

# 🧪 Tool Memory Route
@app.post("/tool/memory")
async def test_route_memory_tool(request: ToolMemoryRequest):
    logging.info(f"🔧 /tool/memory | session={request.session_id}")
    try:
        from memory.memory_router import MemoryRouter
        
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

# ✅ Mount Routers - import order matters!
from api.routes import memory, agent
from api.routes.tool import tool_router  # ✅ NEW

# Include routers
app.include_router(memory.router, prefix="")
app.include_router(agent.router, prefix="")
app.include_router(tool_router, prefix="")  # ✅ NEW

@app.on_event("startup")
async def startup_event():
    logging.info("🚀 FastAPI server started")
    
    memory.init_dependencies()
    agent.init_memory_handler()
    
    for route in app.routes:
        logging.info(f"Registered route: {route.path} - {route.methods}")
