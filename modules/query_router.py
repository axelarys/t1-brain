import os
import psycopg2
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import logging
from typing import List, Union
from config.settings import (
    PG_HOST, PG_DATABASE, PG_USER, PG_PASSWORD,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OPENAI_API_KEY
)
from memory.session_memory import HybridSessionMemory

# Ensure logs directory exists
log_dir = "/root/t1-brain/logs/"
os.makedirs(log_dir, exist_ok=True)

# Remove existing handlers to avoid duplicate logs
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure Logging (File + Console)
log_file = os.path.join(log_dir, "query_router.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode='a'),  # ✅ Write to file
        logging.StreamHandler()                   # ✅ Print to console
    ]
)

logging.info("🚀 QueryRouter logging initialized successfully.")

class QueryRouter:
    """
    Intelligent Query Router that directs queries between Neo4J (Graph DB) and PostgreSQL (Vector DB),
    using OpenAI's GPT-4 model for automated classification.
    """

    def __init__(self):
        """Initialize database connections, AI model, and memory system."""
        self.pg_conn = None
        self.pg_cursor = None
        self.neo4j_driver = None
        self.llm = None
        self.memory = HybridSessionMemory()

        try:
            # Initialize PostgreSQL connection
            self.pg_conn = psycopg2.connect(
                host=PG_HOST, database=PG_DATABASE,
                user=PG_USER, password=PG_PASSWORD
            )
            self.pg_cursor = self.pg_conn.cursor()
            logging.info("✅ PostgreSQL connection established.")

            # Initialize Neo4J connection
            self.neo4j_driver = GraphDatabase.driver(
                NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            with self.neo4j_driver.session() as session:
                session.run("RETURN 1")
            logging.info("✅ Neo4J connection established.")

            # Initialize OpenAI model
            self.llm = ChatOpenAI(
                model_name="gpt-4", openai_api_key=OPENAI_API_KEY
            )
            logging.info("✅ OpenAI Chat Model initialized.")

        except Exception as e:
            logging.error(f"❌ Error initializing QueryRouter: {e}")
            self.close_connections()
            raise

    def classify_intent(self, user_id: str, user_input: str) -> str:
        """
        Uses OpenAI to classify queries as 'graph' (Neo4J) or 'vector' (pgVector).
        Enhances classification with past interactions for better context awareness.
        """
        if not user_input or not isinstance(user_input, str):
            logging.warning("⚠️ Invalid input type or empty input")
            return "unrecognized"

        session_id = f"user_{user_id}"

        # Retrieve past queries from memory for context
        similar_queries = self.memory.find_similar_queries(user_input, min_similarity=0.85)
        past_context = "\n".join([q["query"] for q in similar_queries]) if similar_queries else ""

        try:
            logging.info(f"🔹 Classifying query: {user_input}")

            messages = [
                SystemMessage(content=(
                    "You are an AI that classifies queries strictly as either 'graph' or 'vector'. "
                    "Graph queries involve structured relationships (e.g., user connections, linked data). "
                    "Vector queries involve textual analysis, similarity searches, and embeddings. "
                    "You must return ONLY 'graph' or 'vector'."
                )),
                HumanMessage(content=f"Context: {past_context}\nClassify this query: {user_input}")
            ]

            response = self.llm.invoke(messages)

            if response and hasattr(response, 'content'):
                classification = response.content.lower().strip().replace("'", "")
                logging.info(f"🔹 Initial Classification: {classification}")

                if classification in ["graph", "vector"]:
                    return classification

                # Strict Reclassification
                logging.warning("⚠️ Query unrecognized. Enforcing stricter AI reclassification...")
                deep_messages = [
                    SystemMessage(content="Strictly classify this query as either 'graph' or 'vector'. No other responses."),
                    HumanMessage(content=f"Query: {user_input}\nPast Context: {past_context}")
                ]

                deep_response = self.llm.invoke(deep_messages)

                if deep_response and hasattr(deep_response, 'content'):
                    deep_classification = deep_response.content.lower().strip().replace("'", "")
                    logging.info(f"🔹 Deep Reclassification: {deep_classification}")

                    if deep_classification in ["graph", "vector"]:
                        return deep_classification

            # Log for manual review if AI is uncertain
            self.log_unrecognized_query(user_input)
            return "unrecognized"

        except Exception as e:
            logging.error(f"⚠️ OpenAI Classification Error: {str(e)}")
            return "unrecognized"

    def log_unrecognized_query(self, query: str):
        """Logs unrecognized queries for human review."""
        try:
            with open("/root/t1-brain/logs/unrecognized_queries.log", "a") as log_file:
                log_file.write(f"{query}\n")
            logging.info(f"⚠️ Unrecognized query logged for review: {query}")
        except Exception as e:
            logging.error(f"❌ Error logging unrecognized query: {str(e)}")

    def close_connections(self):
        """Safely closes all database connections."""
        try:
            if self.pg_cursor: self.pg_cursor.close()
            if self.pg_conn: self.pg_conn.close()
            if self.neo4j_driver: self.neo4j_driver.close()
            logging.info("✅ Database connections closed.")
        except Exception as e:
            logging.error(f"⚠️ Error closing connections: {str(e)}")

    def __del__(self):
        """Ensure connections are closed when the object is deleted."""
        self.close_connections()

# Example Usage
if __name__ == "__main__":
    router = QueryRouter()
    test_queries = [
        "Find relationships between users",
        "Find similar research papers about artificial intelligence",
        "Show me transactions connected to a specific user",
        "Analyze the similarities between two articles",
        "How many users have made transactions?",
        "Retrieve connections between employees in an organization",
        "Find outliers in transaction data",
        "Get sentiment analysis for customer reviews"
    ]

    print("\n🔹 Running Query Classification Tests...\n")

    for query in test_queries:
        classification = router.classify_intent("user_123", query)
        print(f"Query: {query} | Classified as: {classification}")

    print("\n✅ Testing Completed!\n")
