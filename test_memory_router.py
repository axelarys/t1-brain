from memory.memory_router import MemoryRouter

router = MemoryRouter()
session_id = "user_123"

test_queries = [
    "Connect my stress at work with the upcoming launch deadline.",
    "Find similar experiences I've logged about burnout.",
    "Update the last memory I had about the April 10 meeting.",
    "Delete anything I said about being anxious.",
    "What were the last few things I said about project Gemini?",
    "Nonsense gibberish input!"
]

print("\n🔁 Running full memory routing + execution test:\n")

for i, query in enumerate(test_queries, 1):
    print(f"{i:02d}. 🔍 Query: {query}")
    classification = router.enrich_and_classify(user_id="123", user_input=query)
    result = router.execute_action(session_id, query, classification)
    print(f"   → Route: {result.get('route')}")
    print(f"   → Status: {result.get('status')}")
    print(f"   → Message: {result.get('message')}\n")

print("✅ All routes tested.\n")
