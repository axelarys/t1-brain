# dev/test_action_parser.py

import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from actions.schema_parser import generate_action_schema

def run_test():
    user_input = "Remind me tomorrow at 8pm to review the RAG whitepaper."
    memory = "You were reading a whitepaper on RAG models yesterday."
    tools = ["set_reminder", "summarize_file", "web_search"]

    result = generate_action_schema(user_input, memory, tools)
    print("\n📦 Parsed Action Schema:")
    print(result)

if __name__ == "__main__":
    run_test()
