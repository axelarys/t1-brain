# graph_memory.py
import logging
from neo4j import GraphDatabase
from config import settings
import traceback

logger = logging.getLogger(__name__)

class GraphMemory:
    """
    Neo4j-based graph memory component for T1 Brain system.
    Manages graph-based storage of queries, responses, and associated metadata.
    """
    def __init__(self):
        try:
            self.driver = GraphDatabase.driver(
                settings.NEO4J_URL,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
                encrypted=False
            )
            if self.verify_connection():
                logger.info("✅ Connected to Neo4j.")
            else:
                logger.warning("⚠️ Connected to Neo4j but verification failed.")
        except Exception as e:
            logger.error(f"❌ Neo4j connection failed: {e}")
            logger.error(traceback.format_exc())
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    def verify_connection(self):
        try:
            if not self.driver:
                logger.error("❌ Neo4j driver is not initialized")
                return False
            with self.driver.session() as session:
                record = session.run("RETURN 1 AS num").single()
                return bool(record and record["num"] == 1)
        except Exception as e:
            logger.error(f"❌ Neo4j connection verification failed: {e}")
            logger.error(traceback.format_exc())
            return False

    def summarize_session(self, session_id: str) -> dict:
        """
        Summarize a session by aggregating all graph-linked metadata.

        Returns:
            {
              "session_id": "<id>",
              "intents": [...],
              "topics": [...],
              "emotions": [...],
              "keywords": [...]
            }
        """
        base = {"session_id": session_id, "intents": [], "topics": [], "emotions": [], "keywords": []}
        if not self.driver:
            logger.error("❌ Cannot summarize session: Neo4j driver is not initialized")
            return base

        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (q:Query {session_id: $session_id, project: 't1brain'})
                    OPTIONAL MATCH (q)-[:HAS_INTENT]->(i:Intent)
                    OPTIONAL MATCH (q)-[:HAS_TOPIC]->(t:Topic)
                    OPTIONAL MATCH (q)-[:EXPRESSES]->(e:Emotion)
                    OPTIONAL MATCH (q)-[:RELATED_TO]->(k:Keyword)
                    RETURN 
                      COLLECT(DISTINCT i.type) AS intents,
                      COLLECT(DISTINCT t.label) AS topics,
                      COLLECT(DISTINCT e.name) AS emotions,
                      COLLECT(DISTINCT k.term) AS keywords
                    """,
                    session_id=session_id
                )
                record = result.single()
                if not record:
                    logger.warning(f"⚠️ No Query nodes found for session: {session_id}")
                    return base

                base["intents"]  = [x for x in record["intents"]  if x]
                base["topics"]   = [x for x in record["topics"]   if x]
                base["emotions"] = [x for x in record["emotions"] if x]
                base["keywords"] = [x for x in record["keywords"] if x]

                logger.info(f"ℹ️ Summarized session {session_id}: "
                            f"{len(base['intents'])} intents, "
                            f"{len(base['topics'])} topics, "
                            f"{len(base['emotions'])} emotions, "
                            f"{len(base['keywords'])} keywords")
                return base

        except Exception as e:
            logger.error(f"❌ Error in summarize_session: {e}")
            logger.error(traceback.format_exc())
            return base

    def add_context_nodes(self, query, response=None, intent=None, topic=None, emotion=None, entities=None, context_props=None):
        """
        Enrichment method to add extended context nodes to Neo4j.
        """
        if not self.driver:
            logger.error("❌ Cannot add context nodes: Neo4j driver is not initialized")
            return False
        if not query:
            logger.error("❌ Cannot add context nodes: query is required")
            return False

        logger.info(f"🚀 Enriching Query: {query[:50]} | intent={intent}, topic={topic}, emotion={emotion}")
        try:
            safe_context = {}
            if context_props:
                safe_context = context_props.copy()
                safe_context.pop("query", None)

            with self.driver.session() as session:
                with session.begin_transaction() as tx:
                    tx.run(
                        """
                        MERGE (q:Query {text: $query_text, project: 't1brain'})
                        SET q += $props
                        """,
                        query_text=query,
                        props=safe_context
                    )
                if response:
                    with session.begin_transaction() as tx:
                        tx.run(
                            """
                            MERGE (r:Response {text: $response, project: 't1brain'})
                            WITH r
                            MATCH (q:Query {text: $query_text, project: 't1brain'})
                            MERGE (q)-[:REPLIED_WITH]->(r)
                            """,
                            query_text=query,
                            response=response
                        )
                if intent:
                    with session.begin_transaction() as tx:
                        tx.run(
                            """
                            MERGE (i:Intent {type: $intent, project: 't1brain'})
                            WITH i
                            MATCH (q:Query {text: $query_text, project: 't1brain'})
                            MERGE (q)-[:HAS_INTENT]->(i)
                            """,
                            query_text=query,
                            intent=intent
                        )
                if topic:
                    with session.begin_transaction() as tx:
                        tx.run(
                            """
                            MERGE (t:Topic {label: $topic, project: 't1brain'})
                            WITH t
                            MATCH (q:Query {text: $query_text, project: 't1brain'})
                            MERGE (q)-[:HAS_TOPIC]->(t)
                            """,
                            query_text=query,
                            topic=topic
                        )
                if emotion:
                    with session.begin_transaction() as tx:
                        tx.run(
                            """
                            MERGE (e:Emotion {name: $emotion, project: 't1brain'})
                            WITH e
                            MATCH (q:Query {text: $query_text, project: 't1brain'})
                            MERGE (q)-[:EXPRESSES]->(e)
                            """,
                            query_text=query,
                            emotion=emotion
                        )
                if entities and isinstance(entities, list):
                    for ent in entities:
                        if ent:
                            with session.begin_transaction() as tx:
                                tx.run(
                                    """
                                    MERGE (ent:Entity {name: $entity, project: 't1brain'})
                                    WITH ent
                                    MATCH (q:Query {text: $query_text, project: 't1brain'})
                                    MERGE (q)-[:MENTIONS]->(ent)
                                    """,
                                    query_text=query,
                                    entity=ent
                                )
            return True

        except Exception as e:
            logger.error(f"❌ Error in add_context_nodes(): {type(e).__name__}: {e}")
            logger.error(traceback.format_exc())
            return False

    def add_keyword_relationship(self, query, keyword):
        """
        Create a relationship between a Query node and a Keyword node.
        """
        try:
            with self.driver.session() as session:
                with session.begin_transaction() as tx:
                    tx.run(
                        """
                        MATCH (q:Query {text: $query_text, project: 't1brain'})
                        MERGE (k:Keyword {term: $keyword, project: 't1brain'})
                        MERGE (q)-[:RELATED_TO]->(k)
                        """,
                        query_text=query,
                        keyword=keyword
                    )
            return True
        except Exception as e:
            logger.error(f"❌ Error in add_keyword_relationship: {e}")
            logger.error(traceback.format_exc())
            return False

    def find_related_contexts(self, query):
        """
        Find relationships and connected context nodes for a given query.
        """
        if not self.driver:
            logger.error("❌ Cannot find contexts: Neo4j driver is not initialized")
            return []
        try:
            with self.driver.session() as session:
                node_res = session.run(
                    "MATCH (q:Query {text: $query_text, project: 't1brain'}) RETURN count(q) AS count",
                    query_text=query
                )
                if node_res.single().get("count", 0) == 0:
                    logger.warning(f"❓ No Query node found for: {query}")
                    return []

                result = session.run(
                    """
                    MATCH (q:Query {text: $query_text, project: 't1brain'})-[r]->(n)
                    RETURN type(r) AS rel_type, labels(n)[0] AS label, properties(n) AS props
                    """,
                    query_text=query
                )
                records = result.data()
                if not records:
                    logger.warning(f"⚠️ Query node exists but no relationships found for: {query}")
                    return []

                return [
                    {"rel_type": r["rel_type"], "label": r["label"], "properties": r["props"]}
                    for r in records
                ]

        except Exception as e:
            logger.error(f"❌ Error in find_related_contexts: {e}")
            logger.error(traceback.format_exc())
            return []

    def debug_enrichment(self, query, metadata=None):
        """
        Debug method to test enrichment functionality directly.
        """
        logger.info(f"🔍 Debug enrichment for query: {query[:50]}")
        logger.info(f"Metadata: {metadata}")
        if not metadata:
            logger.warning("No metadata provided for debug enrichment")
            return False
        intent = metadata.get("intent")
        topic = metadata.get("topic")
        emotion = metadata.get("emotion") or metadata.get("sentiment")
        result = self.add_context_nodes(
            query,
            response=metadata.get("response"),
            intent=intent,
            topic=topic,
            emotion=emotion,
            entities=metadata.get("entities"),
            context_props=metadata.get("context_props")
        )
        logger.info(f"Debug enrichment result: {result}")
        return result

if __name__ == "__main__":
    import sys, argparse
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler()]
    )
    parser = argparse.ArgumentParser(description="T1 Brain Graph Memory Test Tool")
    parser.add_argument("--query", "-q", required=True, help="Test query text")
    parser.add_argument("--intent", "-i", help="Intent classification")
    parser.add_argument("--topic", "-t", help="Topic classification")
    parser.add_argument("--emotion", "-e", help="Emotion classification")
    args = parser.parse_args()
    metadata = {}
    if args.intent: metadata["intent"] = args.intent
    if args.topic:  metadata["topic"]  = args.topic
    if args.emotion:metadata["emotion"] = args.emotion
    logger.info("Initializing GraphMemory...")
    mem = GraphMemory()
    if not mem.verify_connection():
        logger.error("Failed to connect to Neo4j. Check your settings and Neo4j status.")
        sys.exit(1)
    logger.info("Running debug enrichment...")
    ok = mem.debug_enrichment(args.query, metadata)
    sys.exit(0 if ok else 1)
