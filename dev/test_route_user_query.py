# test_route_user_query.py
# ✅ Direct test for MemoryRouter.route_user_query()

import sys
import os
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from memory.memory_router import MemoryRouter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def run_test():
    session_id = "TEST_ROUTER_DIRECT"
    user_input = "DoWebSearch"  # Try "Hello there" for fallback path

    router = MemoryRouter()

    try:
        result = router.route_user_query(session_id=session_id, user_input=user_input)
        logging.info("✅ Result:")
        logging.info(result)
    except Exception as e:
        logging.error(f"❌ Exception: {type(e).__name__}: {e}")

if __name__ == "__main__":
    run_test()
