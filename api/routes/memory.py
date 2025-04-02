from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import Optional

from memory.session_memory import PersistentSessionMemory
from memory.graph_memory import GraphMemory
from memory.memory_router import MemoryRouter

# 🔐 API Key Middleware
def verify_api_key(request: Request):
    api_key = request.headers.get("X-API-KEY")
    if not api_key or api_key != "WsRocks1234":
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return api_key

# 📦 Models
class MemoryRequest(BaseModel):
    session_id: str
    query: str
    response: Optional[str] = None
    memory_type: Optional[str] = "semantic"
    sentiment: Optional[str] = "neutral"

class MemoryDeleteRequest(BaseModel):
    session_id: str
    query: str

class SessionRestoreRequest(BaseModel):
    session_id: str

class MemoryAggregateRequest(BaseModel):
    session_id: str
    query: str

# 📡 Router Setup
router = APIRouter()
memory_handler = PersistentSessionMemory()

# 📥 Store Memory
@router.post("/memory/store")
async def store_memory(request: MemoryRequest, api_key: str = Depends(verify_api_key)):
    return memory_handler.store_memory(
        session_id=request.session_id,
        query=request.query,
        response=request.response,
        memory_type=request.memory_type,
        sentiment=request.sentiment
    )

# 🧠 Retrieve Memory
@router.post("/memory/retrieve")
async def retrieve_memory(request: MemoryRequest, api_key: str = Depends(verify_api_key)):
    raw = memory_handler.retrieve_memory(request.session_id, request.query)
    results = []
    for m in raw:
        entry = {
            "query": m.get("query"),
            "response": m.get("response"),
            "sentiment": m.get("sentiment"),
            "memory_type": m.get("memory_type"),
            "timestamp": m.get("timestamp"),
        }
        if m.get("source_type") == "image":
            entry["source_type"] = "image"
            entry["image_url"] = m.get("image_url")
        results.append(entry)
    return {"status": "retrieved", "count": len(results), "memory": results}

# ♻️ Restore Session from PostgreSQL
@router.post("/memory/session")
async def restore_session(request: SessionRestoreRequest, api_key: str = Depends(verify_api_key)):
    restored = memory_handler.restore_session_from_pg(request.session_id)
    return {"status": "restored", "restored": len(restored), "chunks": restored}

# 🔁 Aggregate Memory (Redis + Vector + Graph)
@router.post("/memory/aggregate")
async def aggregate_memory(request: MemoryAggregateRequest, api_key: str = Depends(verify_api_key)):
    redis_mem = memory_handler.retrieve_memory(request.session_id, request.query)
    vector_matches = memory_handler.find_similar_queries(request.query)
    graph_matches = GraphMemory().retrieve_graph_memory(request.query, top_k=5)
    return {
        "status": "aggregated",
        "redis": redis_mem,
        "vector": vector_matches,
        "graph": graph_matches
    }

# ❌ Delete Memory
@router.delete("/memory/delete")
async def delete_memory(request: MemoryDeleteRequest, api_key: str = Depends(verify_api_key)):
    return memory_handler.delete_memory(request.session_id, request.query)
