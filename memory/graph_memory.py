from neo4j import GraphDatabase
import logging
from datetime import datetime

# Load config
from config.settings import GRAPH_URI, GRAPH_USER, GRAPH_PASSWORD

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

    def store_graph_memory(self, session_id, query_param, response, memory_type, sentiment):
        try:
            with self.driver.session() as session:
                timestamp = datetime.utcnow().isoformat()
                project = "t1brain"

                # Session node
                session.run("""
                    MERGE (s:Session {id: $session_id, project: $project})
                    SET s.last_interaction = $timestamp
                """, session_id=session_id, timestamp=timestamp, project=project)

                # Memory node
                session.run("""
                    CREATE (m:Memory {
                        query_text: $query_param,
                        response: $response,
                        sentiment: $sentiment,
                        memory_type: $memory_type,
                        timestamp: $timestamp,
                        project: $project
                    })
                """, query_param=query_param, response=response,
                     sentiment=sentiment, memory_type=memory_type,
                     timestamp=timestamp, project=project)

                # Link session to memory
                session.run("""
                    MATCH (s:Session {id: $session_id, project: $project})
                    MATCH (m:Memory {query_text: $query_param, timestamp: $timestamp, project: $project})
                    MERGE (s)-[:STORED]->(m)
                """, session_id=session_id, query_param=query_param,
                     timestamp=timestamp, project=project)

                # Emotion node
                session.run("""
                    MERGE (e:Emotion {name: $sentiment, project: $project})
                    WITH e
                    MATCH (m:Memory {query_text: $query_param, timestamp: $timestamp, project: $project})
                    MERGE (m)-[:EXPRESSES]->(e)
                """, sentiment=sentiment, query_param=query_param,
                     timestamp=timestamp, project=project)

                # Basic topic extraction
                topic = " ".join(query_param.strip().split()[:2]) or "unknown"
                session.run("""
                    MERGE (t:Topic {label: $topic, project: $project})
                    WITH t
                    MATCH (m:Memory {query_text: $query_param, timestamp: $timestamp, project: $project})
                    MERGE (m)-[:ABOUT]->(t)
                """, topic=topic, query_param=query_param,
                     timestamp=timestamp, project=project)

        except Exception as e:
            logger.error(f"❌ Error storing to graph DB: {e}")
