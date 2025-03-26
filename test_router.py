from modules.query_router import QueryRouter

router = QueryRouter()

queries = [
    "Find all related nodes for user 123",
    "Find the most similar document to this text: AI and deep learning",
    "Retrieve user details from the database",
    "What’s the latest news about ChatGPT?"
]

for query in queries:
    result = router.classify_intent(query)
    print(f"Query: {query} → Classified as: {result}")

router.close_connections()
