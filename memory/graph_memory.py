from neo4j import GraphDatabase
import logging
from datetime import datetime
import openai
import re
import hashlib
import json

from config.settings import GRAPH_URI, GRAPH_USER, GRAPH_PASSWORD
from utils.memory_utils import get_api_key

# Configure OpenAI
openai.api_key = get_api_key("text")

# Logger setup
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

    def _hash_query(self, query_text):
        return hashlib.sha256(query_text.encode('utf-8')).hexdigest()

    def _extract_enrichment(self, query_param):
        prompt = f"""
        Extract topics, intent, emotion and metadata from this message:
        Return JSON like:
        {{
          "topics": ["topic1", "topic2"],
          "intent": "ask_question/reflect/etc",
          "emotion": "neutral/happy/etc",
          "lifespan": "short_term/long_term",
          "priority": "high/medium/low"
        }}

        Message: "{query_param}"
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            content = response.choices[0].message['content']
            match = re.search(r'{.*}', content, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            logger.warning(f"⚠️ GPT enrichment fallback: {e}")

        return {
            "topics": ["general"],
            "intent": "unknown",
            "emotion": "neutral",
            "lifespan": "short_term",
            "priority": "medium"
        }

    def store_graph_memory(self, session_id, query_param, response, memory_type, sentiment):
        try:
            enrich = self._extract_enrichment(query_param)
            timestamp = datetime.utcnow().isoformat()
            project = "t1brain"
            query_hash = self._hash_query(query_param)

            with self.driver.session() as session:
                # Session node
                session.run("""
                    MERGE (s:Session {id: $session_id, project: $project})
                    SET s.last_interaction = $timestamp
                """, session_id=session_id, timestamp=timestamp, project=project)

                # Memory node
                session.run("""
                    MERGE (m:Memory {hash: $query_hash, project: $project})
                    SET m.query_text = $query_param,
                        m.response = $response,
                        m.sentiment = $sentiment,
                        m.memory_type = $memory_type,
                        m.timestamp = $timestamp,
                        m.lifespan = $lifespan,
                        m.priority = $priority
                """, query_hash=query_hash, query_param=query_param, response=response,
                     sentiment=sentiment, memory_type=memory_type, timestamp=timestamp,
                     lifespan=enrich["lifespan"], priority=enrich["priority"], project=project)

                # Session -> Memory
                session.run("""
                    MATCH (s:Session {id: $session_id, project: $project}),
                          (m:Memory {hash: $query_hash, project: $project})
                    MERGE (s)-[:STORED]->(m)
                """, session_id=session_id, query_hash=query_hash, project=project)

                # Intent + Emotion
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

                # Topic relationships
                for topic in enrich["topics"]:
                    topic = topic.strip()
                    if not topic:
                        continue

                    session.run("""
                        MERGE (t:Topic {label: $topic, project: $project})
                        ON CREATE SET t.created_at = $timestamp
                    """, topic=topic, project=project, timestamp=timestamp)

                    session.run("""
                        MATCH (m:Memory {hash: $query_hash, project: $project}),
                              (t:Topic {label: $topic, project: $project})
                        MERGE (m)-[:MENTIONS]->(t)
                    """, topic=topic, query_hash=query_hash, project=project)

                    session.run("""
                        MATCH (s:Session {id: $session_id, project: $project}),
                              (t:Topic {label: $topic, project: $project})
                        MERGE (s)-[:TOUCHED]->(t)
                    """, session_id=session_id, topic=topic, project=project)

                # Image memory
                if memory_type == "image":
                    session.run("""
                        MERGE (img:Image {hash: $query_hash, project: $project})
                        SET img.description = $query_param,
                            img.timestamp = $timestamp
                    """, query_hash=query_hash, query_param=query_param, timestamp=timestamp, project=project)

                    session.run("""
                        MATCH (img:Image {hash: $query_hash, project: $project}),
                              (m:Memory {hash: $query_hash, project: $project})
                        MERGE (m)-[:REPRESENTED_BY]->(img)
                    """, query_hash=query_hash, project=project)

                    session.run("""
                        MATCH (s:Session {id: $session_id, project: $project}),
                              (img:Image {hash: $query_hash, project: $project})
                        MERGE (s)-[:INCLUDES_IMAGE]->(img)
                    """, session_id=session_id, query_hash=query_hash, project=project)

        except Exception as e:
            logger.error(f"❌ Error storing to graph DB: {e}")

    def retrieve_graph_memory(self, query_text, top_k=5):
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (m:Memory {project: 't1brain'})
                    WHERE m.query_text CONTAINS $query_text OR m.memory_type = $query_text
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
