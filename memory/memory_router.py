# memory_router.py

import os
import json
import logging
import psycopg2
from typing import Optional
from neo4j import GraphDatabase
from openai import OpenAI

from config import settings
from utils.memory_utils import get_api_key
from actions.schema_parser import generate_action_schema

# ——————————————————————————————————————————————————————————————————————————————
# Logging Setup
log_dir = "/root/projects/t1-brain/logs/"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "memory_router.log")
token_log_file = os.path.join(log_dir, "token_usage.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file, mode='a'), logging.StreamHandler()]
)

router_logger = logging.getLogger("memory_router")
router_logger.setLevel(logging.INFO)

token_logger = logging.getLogger("token_logger")
token_handler = logging.FileHandler(token_log_file, mode='a')
token_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
token_logger.addHandler(token_handler)
token_logger.setLevel(logging.INFO)

router_logger.info("🚀 MemoryRouter initialized.")

# ——————————————————————————————————————————————————————————————————————————————
class MemoryRouter:
    def __init__(self):
        self._memory = None
        self.client = OpenAI(api_key=get_api_key("text"))

        # PostgreSQL
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
            self.pg_conn = self.pg_cursor = None

        # Neo4j
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

    def get_tool_agent(self, session_id: str):
        from agents.tool_agent import ToolAgent
        return ToolAgent(session_id)

    # ——————————————————————————————————————————————————————————————————————————————
    def enrich_and_classify(self, user_id: str, user_input: str) -> dict:
        session_id = f"user_{user_id}"
        try:
            sims = self.memory.find_similar_queries(user_input)
        except Exception as e:
            router_logger.warning(f"⚠️ find_similar_queries failed: {e}")
            sims = []
        past_context = "\n".join(q.get("query","") for q in sims) or ""

        system_prompt = (
            "You are an AI that analyzes queries and returns a structured JSON with: "
            "intent, emotion, topic, priority, lifespan, and storage_target "
            "(choose from: 'graph', 'vector', 'update_logic', 'delete_logic')."
        )
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system",  "content": system_prompt},
                    {"role": "user",    "content": f"Context: {past_context}\nQuery: {user_input}"}
                ]
            )
            usage = getattr(response, 'usage', None)
            if usage:
                token_logger.info(
                    f"enrichment | session={session_id} | tokens={usage.total_tokens} | model=gpt-4"
                )
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

    # ——————————————————————————————————————————————————————————————————————————————
    def route_user_query(self, session_id: str, user_input: str) -> dict:
        """
        Full pipeline:
          1) generate schema via GPT → 2) tool vs. none → 3) execute → 4) persist memory → 5) return result
        """
        # 1) Produce a schema
        try:
            sims = self.memory.find_similar_queries(user_input)
        except Exception:
            sims = []
        memory_snippet = "\n".join(m.get("query","") for m in sims)[:500]
        tools = ["set_reminder", "summarize_file", "web_search", "storeMemory", "updateMemory"]
        schema = generate_action_schema(user_input, memory_snippet, tools)

        # 2) Either tool or fallback
        if schema.get("action") == "none":
            router_logger.info("🧠 Routed as fallback (no tool).")
            return self.execute_action(
                session_id=session_id,
                user_input=user_input,
                enriched=schema   # pass the schema (action=none, parameters={})
            )
        else:
            router_logger.info(f"⚙️ Routed as GPT Action: {schema['action']}")
            return self.execute_action(
                session_id=session_id,
                user_input=user_input,
                enriched=schema
            )

    # ——————————————————————————————————————————————————————————————————————————————
    def execute_action(
        self,
        session_id: str,
        user_input: str,
        enriched: Optional[dict] = None
    ) -> dict:
        """
        Executes the 'action' in `enriched`:
          • if action='none' → ToolAgent.fallback
          • else → ToolAgent.run_single_tool
        Then persists the result into memory.
        """
        # 0) If enriched is malformed, reclassify
        if not enriched or "action" not in enriched:
            enriched = self.enrich_and_classify(session_id, user_input)

        try:
            agent  = self.get_tool_agent(session_id)
            result = agent.run(enriched)
            output = result.get("output", "")

            # 3) Decide memory type
            mem_type = (
                "tool" if result.get("status")=="success" and result.get("tool")!="none"
                else "semantic"
            )

            # 4) Persist memory
            try:
                # You can extend metadata here with intent/topic/emotion if desired
                self.memory.store_memory(
                    session_id=session_id,
                    query=user_input,
                    response=str(output),
                    memory_type=mem_type,
                    sentiment=result.get("emotion","neutral"),
                    metadata={"tool": result.get("tool","")}
                )
                router_logger.info(f"✅ Stored '{mem_type}' memory for session={session_id}")
            except Exception as e:
                router_logger.warning(f"⚠️ Failed to store action memory: {e}")

            return result

        except Exception as e:
            router_logger.error(f"❌ Action execution failed: {e}", exc_info=True)
            return {"status":"error", "message":"Tool execution failed"}
