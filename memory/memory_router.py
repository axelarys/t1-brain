import os
import json
import logging
import psycopg2
from neo4j import GraphDatabase
from config import settings
from openai import OpenAI
from memory.session_memory import PersistentSessionMemory

# Setup logging
log_dir = "/root/t1-brain/logs/"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "memory_router.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode='a'),
        logging.StreamHandler()
    ]
)
logging.info("🚀 MemoryRouter initialized.")

class MemoryRouter:
    def __init__(self):
        self.memory = PersistentSessionMemory()
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

        try:
            self.pg_conn = psycopg2.connect(
                host=settings.PG_HOST,
                database=settings.PG_DATABASE,
                user=settings.PG_USER,
                password=settings.PG_PASSWORD
            )
            self.pg_cursor = self.pg_conn.cursor()
            logging.info("✅ PostgreSQL connection established.")
        except Exception as e:
            logging.error(f"❌ PostgreSQL connection error: {e}")
            self.pg_conn, self.pg_cursor = None, None

        try:
            self.neo4j_driver = GraphDatabase.driver(
                settings.GRAPH_URI,
                auth=(settings.GRAPH_USER, settings.GRAPH_PASSWORD)
            )
            with self.neo4j_driver.session() as session:
                session.run("RETURN 1")
            logging.info("✅ Neo4j connection established.")
        except Exception as e:
            logging.error(f"❌ Neo4j connection error: {e}")
            self.neo4j_driver = None

    def enrich_and_classify(self, user_id: str, user_input: str) -> dict:
        """Enrich input using OpenAI and classify routing target."""
        session_id = f"user_{user_id}"
        past_context = "\n".join([q["query"] for q in self.memory.find_similar_queries(user_input)]) or ""

        prompt = (
            "You are an AI that analyzes queries and returns a structured JSON with: "
            "intent, emotion, topic, priority, lifespan, and storage_target "
            "(choose one of: 'graph', 'vector', 'update_logic', 'delete_logic')."
        )
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Context: {past_context}\nQuery: {user_input}"}
                ]
            )
            content = response.choices[0].message.content
            logging.info(f"🧠 Enrichment Output:\n{content}")
            return self._parse_enrichment(content, user_input)
        except Exception as e:
            logging.error(f"❌ OpenAI enrichment error: {e}")
            return self._fallback_classification(user_input)

    def _parse_enrichment(self, raw_output: str, original_query: str) -> dict:
        try:
            start = raw_output.find('{')
            end = raw_output.rfind('}') + 1
            json_text = raw_output[start:end]
            enriched = json.loads(json_text)

            if enriched.get("storage_target") not in ["graph", "vector", "update_logic", "delete_logic"]:
                enriched["storage_target"] = self._fallback_classification(original_query)["storage_target"]

            return enriched
        except Exception as e:
            logging.warning(f"⚠️ Failed to parse enrichment JSON: {e}")
            return self._fallback_classification(original_query)

    def _fallback_classification(self, user_input: str) -> dict:
        query = user_input.lower()
        if any(k in query for k in ["find", "similar", "summarize", "search", "analyze"]):
            route = "vector"
        elif any(k in query for k in ["connect", "relationship", "link", "between", "graph"]):
            route = "graph"
        elif "update" in query:
            route = "update_logic"
        elif "delete" in query:
            route = "delete_logic"
        else:
            route = "vector"

        logging.info(f"🔁 Fallback classification used: {route}")
        return {
            "intent": "fallback",
            "emotion": "neutral",
            "topic": "unknown",
            "priority": "low",
            "lifespan": "session",
            "storage_target": route
        }

    def execute_action(self, session_id: str, user_input: str, classification: dict):
        route = classification.get("storage_target")
        intent = classification.get("intent", "")
        response_text = ""

        if route == "vector":
            matches = self.memory.find_similar_queries(user_input)
            response_text = f"🔎 Retrieved {len(matches)} similar memory entries from vector DB."
            return {
                "status": "retrieved",
                "route": route,
                "matches": matches,
                "message": response_text
            }

        elif route == "graph":
            self.memory.store_memory(session_id, user_input, response="Stored via graph logic",
                                     memory_type="semantic", sentiment=classification.get("emotion", "neutral"))
            return {
                "status": "stored",
                "route": route,
                "message": "Memory stored in Graph DB (Neo4j)."
            }

        elif route == "update_logic":
            return {
                "status": "update_pending",
                "route": route,
                "message": "Memory update functionality triggered (placeholder)."
            }

        elif route == "delete_logic":
            self.memory.delete_memory(session_id, user_input)
            return {
                "status": "deleted",
                "route": route,
                "message": "Memory deleted from Redis and PostgreSQL."
            }

        else:
            return {
                "status": "fallback",
                "route": "vector",
                "message": "Fallback route executed. Treated as vector query."
            }

    def close_connections(self):
        if self.pg_cursor: self.pg_cursor.close()
        if self.pg_conn: self.pg_conn.close()
        if self.neo4j_driver: self.neo4j_driver.close()
        logging.info("🔌 Connections closed.")

    def __del__(self):
        self.close_connections()
