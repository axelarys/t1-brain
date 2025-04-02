from fastapi import APIRouter, Depends, HTTPException, Request
from typing import Optional

# Import shared models 
from api.models import (
    MemoryRequest, 
    MemoryDeleteRequest, 
    SessionRestoreRequest, 
    MemoryAggregateRequest
)

# 📡 Router Setup
router = APIRouter()

# 🔐 API Key Middleware
def verify_api_key(request: Request):
    api_key = request.headers.get("X-API-KEY")
    if not api_key or api_key != "WsRocks1234":
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return api_key

# Dependency providers
_memory_handler = None
_graph_memory = None

def init_dependencies():
    """Initialize required dependencies - called during app startup"""
    global _memory_handler, _graph_memory
    
    # Import here to avoid circular import
    from memory.session_memory import PersistentSessionMemory
    from memory.graph_memory import GraphMemory
    
    if _memory_handler is None:
        _memory_handler = PersistentSessionMemory()
    
    if _graph_memory is None:
        _graph_memory = GraphMemory()

def get_memory_handler():
    if _memory_handler is None:
        raise HTTPException(status_code=500, detail="Memory handler not initialized")
    return _memory_handler

def get_graph_memory():
    if _graph_memory is None:
        raise HTTPException(status_code=500, detail="Graph memory not initialized")
    return _graph_memory

# 📥 Store Memory
@router.post("/memory/store")
async def store_memory(
    request: MemoryRequest, 
    api_key: str = Depends(verify_api_key)
):
    memory_handler = get_memory_handler()
    return memory_handler.store_memory(
        session_id=request.session_id,
        query=request.query,
        response=request.response,
        memory_type=request.memory_type,
        sentiment=request.sentiment
    )

# 🧠 Retrieve Memory
@router.post("/memory/retrieve")
async def retrieve_memory(
    request: MemoryRequest, 
    api_key: str = Depends(verify_api_key)
):
    memory_handler = get_memory_handler()
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
async def restore_session(
    request: SessionRestoreRequest, 
    api_key: str = Depends(verify_api_key)
):
    memory_handler = get_memory_handler()
    restored = memory_handler.restore_session_from_pg(request.session_id)
    return {"status": "restored", "restored": len(restored), "chunks": restored}

# 🔁 Aggregate Memory (Redis + Vector + Graph)
@router.post("/memory/aggregate")
async def aggregate_memory(
    request: MemoryAggregateRequest, 
    api_key: str = Depends(verify_api_key)
):
    memory_handler = get_memory_handler()
    graph_memory = get_graph_memory()
    
    redis_mem = memory_handler.retrieve_memory(request.session_id, request.query)
    vector_matches = memory_handler.find_similar_queries(request.query)
    graph_matches = graph_memory.retrieve_graph_memory(request.query, top_k=5)
    
    return {
        "status": "aggregated",
        "redis": redis_mem,
        "vector": vector_matches,
        "graph": graph_matches
    }

# ❌ Delete Memory
@router.delete("/memory/delete")
async def delete_memory(
    request: MemoryDeleteRequest, 
    api_key: str = Depends(verify_api_key)
):
    memory_handler = get_memory_handler()
    return memory_handler.delete_memory(request.session_id, request.query)