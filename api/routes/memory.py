from fastapi import APIRouter, Request
from pydantic import BaseModel
from memory.memory_router import MemoryRouter

router = APIRouter()
memory_router = MemoryRouter()

class MemoryRouteRequest(BaseModel):
    session_id: str
    user_input: str

@router.post("/memory/route")
async def route_and_execute_memory(request: MemoryRouteRequest):
    try:
        classification = memory_router.enrich_and_classify(
            user_id=request.session_id, user_input=request.user_input
        )
        result = memory_router.execute_action(
            session_id=request.session_id,
            user_input=request.user_input,
            classification=classification
        )
        return {
            "status": result.get("status"),
            "route": result.get("route"),
            "message": result.get("message"),
            "matches": result.get("matches", []),
            "meta": classification
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
