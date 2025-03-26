# test_session_memory.py

import sys
sys.path.append("memory")

from session_memory import PersistentSessionMemory

memory = PersistentSessionMemory()
result = memory.store_memory(
    session_id="test001",
    query="What is AI?",
    response="AI stands for Artificial Intelligence.",
    memory_type="semantic",
    sentiment="neutral"
)

print(result)
