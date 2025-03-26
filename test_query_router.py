import sys
sys.path.append("/root/t1-brain/modules/")  # Ensure correct import path

from query_router import QueryRouter

def test_classification():
    """Run a batch of classification tests for QueryRouter."""
    query_router = QueryRouter()

    queries = {
        "Find relationships between users": "graph",
        "Show me transactions connected to a specific user": "graph",
        "Find similar research papers about artificial intelligence": "vector",
        "What is the sentiment of this text?": "vector",
        "How many users have made transactions?": "graph",
        "Find related topics to machine learning": "vector",
        "Analyze the similarities between two news articles": "vector",
        "Retrieve connections between employees in an organization": "graph"
    }

    print("\n🔹 Running Deep Query Classification Tests...\n")

    for query, expected in queries.items():
        result = query_router.classify_intent("user_123", query)
        print(f"Query: {query}\nExpected: {expected} | Classified as: {result}")
        assert result == expected, f"❌ Classification Error: Expected {expected}, got {result}"

    print("\n✅ All tests passed successfully!\n")

if __name__ == "__main__":
    test_classification()
