# /root/projects/t1-brain/api/models.py
from pydantic import BaseModel
from typing import Optional

# 📦 Memory Request Models
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

class ToolMemoryRequest(BaseModel):
    session_id: str
    user_input: str