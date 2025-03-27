# test_graph_memory.py

from memory.graph_memory import GraphMemory

graph = GraphMemory()

graph.store_graph_memory(
    session_id="graph-test-003",
    query_param="What is the future of AI?",
    response="AI will continue to evolve toward more human-like behavior.",
    memory_type="semantic",
    sentiment="optimistic"
)

print("✅ Graph memory insertion attempted.")
