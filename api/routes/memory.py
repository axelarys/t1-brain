# memory.py
from fastapi import APIRouter, Depends, HTTPException, Request
from typing import Optional

from api.models import (
    MemoryRequest, 
    MemoryDeleteRequest, 
    SessionRestoreRequest, 
    MemoryAggregateRequest
)

router = APIRouter()

def verify_api_key(request: Request):
    api_key = request.headers.get("X-API-KEY")
    if not api_key or api_key != "WsRocks1234":
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return api_key

_memory_handler = None
_graph_memory = None

def init_dependencies():
    """Initialize required dependencies - called during app startup"""
    global _memory_handler, _graph_memory
    
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

@router.post("/memory/store")
async def store_memory(request: MemoryRequest, api_key: str = Depends(verify_api_key)):
    handler = get_memory_handler()
    return handler.store_memory(
        session_id=request.session_id,
        query=request.query,
        response=request.response,
        memory_type=request.memory_type,
        sentiment=request.sentiment
    )

@router.post("/memory/retrieve")
async def retrieve_memory(request: MemoryRequest, api_key: str = Depends(verify_api_key)):
    handler = get_memory_handler()
    raw = handler.retrieve_memory(request.session_id, request.query)
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

@router.post("/memory/session")
async def restore_session(request: SessionRestoreRequest, api_key: str = Depends(verify_api_key)):
    handler = get_memory_handler()
    restored = handler.restore_session_from_pg(request.session_id)
    return {"status": "restored", "restored": len(restored), "chunks": restored}

@router.post("/memory/aggregate")
async def aggregate_memory(request: MemoryAggregateRequest, api_key: str = Depends(verify_api_key)):
    handler = get_memory_handler()
    graph = get_graph_memory()
    return {
        "status": "aggregated",
        "redis": handler.retrieve_memory(request.session_id, request.query),
        "vector": handler.find_similar_queries(request.query),
        "graph": graph.retrieve_graph_memory(request.query, top_k=5)
    }

@router.delete("/memory/delete")
async def delete_memory(request: MemoryDeleteRequest, api_key: str = Depends(verify_api_key)):
    handler = get_memory_handler()
    return handler.delete_memory(request.session_id, request.query)

@router.get("/memory/summarize")
async def summarize_session(session_id: str, api_key: str = Depends(verify_api_key)):
    handler = get_memory_handler()
    return handler.summarize_session(session_id)

@router.post("/memory/cleanup")
async def cleanup_session_memory(session_id: str, api_key: str = Depends(verify_api_key)):
    handler = get_memory_handler()
    return handler.cleanup_expired_memory(session_id)
