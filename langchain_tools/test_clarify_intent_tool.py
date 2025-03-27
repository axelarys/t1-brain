import sys
sys.path.append("/root/projects/t1-brain/langchain_tools")

from clarify_intent_tool import clarify_intent

# Simulate ambiguous input
response = clarify_intent.invoke({
    "user_input": "Can you make this better?"
})

print("🧠 Clarifier:", response)
