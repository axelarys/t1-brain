import os, json, logging, psycopg2
from neo4j import GraphDatabase
from openai import OpenAI

from config import settings
from utils.memory_utils import get_api_key
from actions.schema_parser import generate_action_schema

# 📁 Logging Setup
log_dir = "/root/projects/t1-brain/logs/"
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, "memory_router.log")
token_log_file = os.path.join(log_dir, "token_usage.log")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file, mode='a'), logging.StreamHandler()])

router_logger = logging.getLogger("memory_router")
router_logger.setLevel(logging.INFO)

token_logger = logging.getLogger("token_logger")
token_handler = logging.FileHandler(token_log_file, mode='a')
token_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
token_logger.addHandler(token_handler)
token_logger.setLevel(logging.INFO)

router_logger.info("🚀 MemoryRouter initialized.")


class MemoryRouter:
    def __init__(self):
        self._memory = None
        self.client = OpenAI(api_key=get_api_key("text"))

        try:
            self.pg_conn = psycopg2.connect(
                host=settings.PG_HOST,
                database=settings.PG_DATABASE,
                user=settings.PG_USER,
                password=settings.PG_PASSWORD
            )
            self.pg_cursor = self.pg_conn.cursor()
            router_logger.info("✅ PostgreSQL connection established.")
        except Exception as e:
            router_logger.error(f"❌ PostgreSQL connection error: {e}")
            self.pg_conn, self.pg_cursor = None, None

        try:
            self.neo4j_driver = GraphDatabase.driver(
                settings.GRAPH_URI,
                auth=(settings.GRAPH_USER, settings.GRAPH_PASSWORD)
            )
            with self.neo4j_driver.session() as session:
                session.run("RETURN 1")
            router_logger.info("✅ Neo4j connection established.")
        except Exception as e:
            router_logger.error(f"❌ Neo4j connection error: {e}")
            self.neo4j_driver = None
            
    @property
    def memory(self):
        if self._memory is None:
            from memory.session_memory import PersistentSessionMemory
            self._memory = PersistentSessionMemory()
        return self._memory

    def enrich_and_classify(self, user_id: str, user_input: str) -> dict:
        session_id = f"user_{user_id}"
        past_context = "\n".join([q["query"] for q in self.memory.find_similar_queries(user_input)]) or ""

        system_prompt = (
            "You are an AI that analyzes queries and returns a structured JSON with: "
            "intent, emotion, topic, priority, lifespan, and storage_target "
            "(choose from: 'graph', 'vector', 'update_logic', 'delete_logic')."
        )

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context: {past_context}\nQuery: {user_input}"}
                ]
            )
            usage = getattr(response, 'usage', None)
            if usage:
                token_logger.info(f"enrichment | session={session_id} | tokens={usage.total_tokens} | model=gpt-4")

            content = response.choices[0].message.content
            router_logger.info(f"🧠 Enrichment Output:\n{content}")
            return self._parse_enrichment(content, user_input)
        except Exception as e:
            router_logger.error(f"❌ OpenAI enrichment error: {e}")
            return self._fallback_classification(user_input)

    def _parse_enrichment(self, content: str, query: str) -> dict:
        try:
            data = json.loads(content)
            return {
                "query": query,
                "intent": data.get("intent", "unknown"),
                "emotion": data.get("emotion", "neutral"),
                "topic": data.get("topic", "general"),
                "priority": data.get("priority", "medium"),
                "lifespan": data.get("lifespan", "short_term"),
                "storage_target": data.get("storage_target", "graph")
            }
        except Exception as e:
            router_logger.warning(f"⚠️ Failed to parse enrichment, fallback: {e}")
            return self._fallback_classification(query)

    def _fallback_classification(self, query: str) -> dict:
        return {
            "query": query,
            "intent": "unknown",
            "emotion": "neutral",
            "topic": "general",
            "priority": "medium",
            "lifespan": "short_term",
            "storage_target": "graph"
        }

    def route_user_query(self, session_id: str, user_input: str) -> dict:
        memory_snippet = "\n".join([m["query"] for m in self.memory.find_similar_queries(user_input)])[:500]
        tools = ["set_reminder", "summarize_file", "web_search", "storeMemory", "updateMemory"]

        schema = generate_action_schema(user_input, memory_snippet, tools)

        if schema["action"] == "none":
            router_logger.info("🧠 Routed as memory enrichment (fallback).")
            enriched = self.enrich_and_classify(session_id, user_input)
            return self.execute_action(session_id, user_input, enriched)
        else:
            router_logger.info(f"⚙️ Routed as GPT Action: {schema['action']}")
            return {
                "status": "tool_action",
                "action": schema["action"],
                "parameters": schema["parameters"]
            }

    def execute_action(self, session_id: str, user_input: str, enriched: dict) -> dict:
        try:
            response = ""  # Placeholder response
            return self.memory.store_memory(
                session_id=session_id,
                query=user_input,
                response=response,
                memory_type=enriched.get("storage_target", "semantic"),
                sentiment=enriched.get("emotion", "neutral")
            )
        except Exception as e:
            router_logger.error(f"❌ Action execution failed: {e}")
            return {"status": "error", "message": "Failed to execute memory logic"}
