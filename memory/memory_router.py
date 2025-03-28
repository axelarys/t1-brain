# [UNCHANGED IMPORTS]
import os, json, logging, psycopg2
from neo4j import GraphDatabase
from config import settings
from openai import OpenAI
from memory.session_memory import PersistentSessionMemory

# Setup logging
log_dir = "/root/projects/t1-brain/logs/"
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, "memory_router.log")
token_log_file = os.path.join(log_dir, "token_usage.log")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file, mode='a'), logging.StreamHandler()])
logging.info("🚀 MemoryRouter initialized.")

token_logger = logging.getLogger("token_logger")
token_handler = logging.FileHandler(token_log_file, mode='a')
token_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
token_logger.addHandler(token_handler)
token_logger.setLevel(logging.INFO)


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
        session_id = f"user_{user_id}"
        past_context = "\n".join([q["query"] for q in self.memory.find_similar_queries(user_input)]) or ""

        prompt = (
            "You are an AI that analyzes queries and returns a structured JSON with: "
            "intent, emotion, topic, priority, lifespan, and storage_target "
            "(choose from: 'graph', 'vector', 'update_logic', 'delete_logic')."
        )

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Context: {past_context}\nQuery: {user_input}"}
                ]
            )
            usage = getattr(response, 'usage', None)
            if usage:
                token_logger.info(f"enrichment | session={session_id} | tokens={usage.total_tokens} | model=gpt-4")

            content = response.choices[0].message.content
            logging.info(f"🧠 Enrichment Output:\n{content}")
            return self._parse_enrichment(content, user_input)
        except Exception as e:
            logging.error(f"❌ OpenAI enrichment error: {e}")
            return self._fallback_classification(user_input)

# [REST OF FILE UNCHANGED... execute_action() etc.]
