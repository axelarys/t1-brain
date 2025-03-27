# test_session_memory.py

import sys
sys.path.append("memory")
from session_memory import PersistentSessionMemory

memory = PersistentSessionMemory()
result = memory.store_memory(
    session_id="graph-test-002",
    query="What is the future of AI?",
    response="AI is expected to become more human-like and context-aware.",
    memory_type="semantic",
    sentiment="curious"
)

print(result)
