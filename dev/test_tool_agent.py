# dev/test_tool_agent.py

import sys
sys.path.append("/root/projects/t1-brain")

from agents.tool_agent import ToolAgent

test_action = {
    "action": "set_reminder",
    "parameters": {
        "task": "submit quarterly report",
        "date": "next Monday",
        "time": "9am"
    }
}

agent = ToolAgent(session_id="user_agent_test")
result = agent.run(test_action)

print("📦 Agent Result:")
print(result)
