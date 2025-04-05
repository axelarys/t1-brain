import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memory.memory_router import MemoryRouter  # ✅ Corrected import

def run_test():
    router = MemoryRouter()
    session_id = "user_test"
    user_input = "Remind me to follow up with the investor on Monday at 10am"

    result = router.route_user_query(session_id=session_id, user_input=user_input)

    print("\n📦 Routing Result:")
    print(result)

if __name__ == "__main__":
    run_test()
