import sys
sys.path.append("/root/projects/t1-brain/langchain_tools")

from memory_agent import agent

# 🔍 Try an ambiguous query
response = agent.run("Can you make this better?")

print("\n🤖 Final Agent Response:")
print(response)
