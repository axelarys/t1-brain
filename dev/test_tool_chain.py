# dev/test_tool_chain.py

import sys
sys.path.append("/root/projects/t1-brain")

from agents.tool_agent import ToolAgent

agent = ToolAgent(session_id="user_chain_test")

action_chain = {
    "action": "multi_tool_chain",
    "parameters": {
        "chain": [
            {
                "action": "file_reader",
                "parameters": {
                    "file_path": "/root/projects/t1-brain/scb.pdf",
                    "summary": True
                }
            },
            {
                "action": "set_reminder",
                "parameters": {
                    "task": "<result_from_0>",
                    "date": "Friday",
                    "time": "5pm"
                }
            }
        ]
    }
}

result = agent.run(action_chain)

print("🧠 Tool Chain Result:")
print(result)
