import requests
from langchain.tools import tool

API_URL = "http://localhost:8000/memory/route"
API_KEY = "test123"

@tool
def route_memory(session_id: str, user_input: str) -> str:
    """
    Routes a memory query to the T1 Brain memory system. Accepts a session ID and natural language input.
    Returns the result (retrieved, stored, updated, etc.)
    """
    try:
        response = requests.post(
            API_URL,
            headers={"Content-Type": "application/json", "X-API-KEY": API_KEY},
            json={"session_id": session_id, "user_input": user_input}
        )
        if response.status_code == 200:
            data = response.json()
            return f"✅ {data['message']} (Route: {data['route']})"
        else:
            return f"❌ Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"❌ Exception during request: {str(e)}"
