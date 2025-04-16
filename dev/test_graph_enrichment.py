# test_graph_enrichment.py
# ✅ CLI test runner for full memory graph enrichment via session_memory.py

import sys
import os
import logging
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from memory.session_memory import PersistentSessionMemory

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def run_test():
    session_id = f"debug_graph_test_{int(time.time())}"
    query = "How can I visualize memory structures?"
    response = "Use Neo4j to visualize intents, topics, and emotions in memory."

    metadata = {
        "intent": "explore",
        "topic": "graph enrichment",
        "emotion": "curious",
        "entities": ["memory structures", "Neo4j"],
        "keywords": ["visualization", "intent", "topic", "emotion"]
    }

    logger.info(f"🚀 Initializing PersistentSessionMemory... (session_id={session_id})")
    memory = PersistentSessionMemory()

    logger.info("🧪 Storing test memory with enrichment...")
    success = memory.store_memory(
        session_id=session_id,
        query=query,
        response=response,
        memory_type="semantic",
        sentiment="curious",
        source_type="text",
        metadata=metadata
    )

    if success:
        logger.info("✅ Test memory stored and graph enrichment completed")
    else:
        logger.error("❌ Test failed: memory storage or enrichment unsuccessful")

if __name__ == "__main__":
    run_test()