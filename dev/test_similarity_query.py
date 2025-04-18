# test_similarity_query.py

import os
import sys
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from memory.session_memory import PersistentSessionMemory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)

def run_test():
    mem = PersistentSessionMemory()
    test_query = "I love pie"
    logger.info(f"🔍 Running similarity test on query: {test_query}")

    results = mem.find_similar_queries(test_query, top_k=3)

    logger.info("✅ Result:")
    for idx, r in enumerate(results):
        logger.info(f"{idx+1}. {r.get('query', '<no query>')}")

    if not results:
        logger.warning("⚠️ No similar results found. Warm cache may be empty.")

if __name__ == "__main__":
    run_test()
