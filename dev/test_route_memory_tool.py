import sys
sys.path.append("/root/projects/t1-brain/langchain_tools")

from route_memory_tool import route_memory

# Simulated input
result = route_memory.invoke({
    "session_id": "user_tool_001",
    "user_input": "Remind me what I said about project deadlines."
})

print(result)
