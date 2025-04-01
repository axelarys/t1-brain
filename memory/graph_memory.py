from neo4j import GraphDatabase
import logging
from datetime import datetime
import openai
import re
import hashlib

from config.settings import GRAPH_URI, GRAPH_USER, GRAPH_PASSWORD
from utils.memory_utils import get_api_key

# Configure OpenAI using dynamic API key
openai.api_key = get_api_key("text")

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GraphMemory:
    def __init__(self):
        try:
            self.driver = GraphDatabase.driver(GRAPH_URI, auth=(GRAPH_USER, GRAPH_PASSWORD))
            logger.info("✅ Connected to Neo4j.")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Neo4j: {e}")
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    def _extract_enrichment(self, query_param):
        """Use OpenAI to extract emotion, intent, topic, lifespan, priority, reusability."""
        prompt = f"""
        Analyze the user's message and return:
        {{
          "emotion": "happy/sad/etc",
          "intent": "ask_question/reflect/etc",
          "topic": "short_label",
          "lifespan": "short_term/long_term/one_time",
          "priority": "high/medium/low",
          "reusability": "yes/no"
        }}

        Message: "{query_param}"
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4
            )
            text = response.choices[0].message['content']
            match = re.search(r'{.*}', text, re.DOTALL)
            if match:
                data = eval(match.group())
                return {
                    "emotion": data.get("emotion", "neutral"),
                    "intent": data.get("intent", "unknown"),
                    "topic": data.get("topic", "general"),
                    "lifespan": data.get("lifespan", "short_term"),
                    "priority": data.get("priority", "medium"),
                    "reusability": data.get("reusability", "no")
                }
        except Exception as e:
            logger.error(f"❌ OpenAI enrichment failed: {e}")

        return {
            "emotion": "neutral",
            "intent": "unknown",
            "topic": "general",
            "lifespan": "short_term",
            "priority": "medium",
            "reusability": "no"
        }

    def _hash_query(self, query_text):
        return hashlib.sha256(query_text.encode('utf-8')).hexdigest()

    def store_graph_memory(self, session_id, query_param, response, memory_type, sentiment):
        """Store enriched memory into graph DB with auto property handling."""
        try:
            enrich = self._extract_enrichment(query_param)
            timestamp = datetime.utcnow().isoformat()
            project = "t1brain"
            query_hash = self._hash_query(query_param)

            with self.driver.session() as session:
                # Session Node
                session.run("""
                    MERGE (s:Session {id: $session_id, project: $project})
                    SET s.last_interaction = $timestamp
                """, session_id=session_id, timestamp=timestamp, project=project)

                # Memory Node
                session.run("""
                    MERGE (m:Memory {hash: $query_hash, project: $project})
                    SET m.query_text = $query_param,
                        m.response = $response,
                        m.sentiment = $sentiment,
                        m.memory_type = $memory_type,
                        m.timestamp = $timestamp,
                        m.lifespan = $lifespan,
                        m.priority = $priority,
                        m.reusability = $reusability,
                        m.topic = $topic
                """, query_hash=query_hash, query_param=query_param, response=response,
                     sentiment=sentiment, memory_type=memory_type,
                     timestamp=timestamp, project=project,
                     lifespan=enrich["lifespan"], priority=enrich["priority"],
                     reusability=enrich["reusability"], topic=enrich["topic"])

                # Relationships
                session.run("""
                    MATCH (s:Session {id: $session_id, project: $project}),
                          (m:Memory {hash: $query_hash, project: $project})
                    MERGE (s)-[:STORED]->(m)
                """, session_id=session_id, query_hash=query_hash, project=project)

                session.run("""
                    MERGE (e:Emotion {name: $emotion, project: $project})
                    WITH e
                    MATCH (m:Memory {hash: $query_hash, project: $project})
                    MERGE (m)-[:EXPRESSES]->(e)
                """, emotion=enrich["emotion"], query_hash=query_hash, project=project)

                session.run("""
                    MERGE (i:Intent {type: $intent, project: $project})
                    WITH i
                    MATCH (m:Memory {hash: $query_hash, project: $project})
                    MERGE (m)-[:HAS_INTENT]->(i)
                """, intent=enrich["intent"], query_hash=query_hash, project=project)

                session.run("""
                    MERGE (t:Topic {label: $topic, project: $project})
                    WITH t
                    MATCH (m:Memory {hash: $query_hash, project: $project})
                    MERGE (m)-[:ABOUT]->(t)
                """, topic=enrich["topic"], query_hash=query_hash, project=project)

        except Exception as e:
            logger.error(f"❌ Error storing to graph DB: {e}")

    def retrieve_graph_memory(self, query_text, top_k=5):
        """Retrieve related memory nodes."""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (m:Memory {project: 't1brain'})
                    WHERE m.query_text CONTAINS $query_text OR m.topic CONTAINS $query_text
                    RETURN m.query_text AS query, m.response AS response, m.timestamp AS timestamp,
                           m.sentiment AS sentiment, m.memory_type AS memory_type,
                           m.priority AS priority, m.lifespan AS lifespan
                    ORDER BY m.timestamp DESC
                    LIMIT $top_k
                """, query_text=query_text, top_k=top_k)

                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"❌ Error retrieving from graph DB: {e}")
            return []
