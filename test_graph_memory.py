# test_graph_memory.py

import sys
sys.path.append("/root/projects/t1-brain/memory")

from memory.graph_memory import GraphMemory

graph = GraphMemory()
query = "future of artificial intelligence"
results = graph.retrieve_graph_memory(query)

for item in results:
    print(item)

