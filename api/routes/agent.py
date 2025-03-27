from fastapi import APIRouter
from pydantic import BaseModel
from langchain_tools.memory_agent import agent

router = APIRouter()

class AgentInput(BaseModel):
    session_id: str
    user_input: str

@router.post("/agent/run")
async def run_agent(request: AgentInput):
    try:
        # Let agent dynamically decide action
        result = agent.invoke(f"[{request.session_id}] {request.user_input}")
        return {
            "status": "success",
            "response": result
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
