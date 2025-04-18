# session_memory.py

import os
import json
import time
import redis
import logging
import psycopg2
import hashlib
import numpy as np
import tiktoken
import traceback
from datetime import datetime

from config import settings
from openai import OpenAI
from utils.memory_utils import get_api_key
from memory.warm_layer import WarmMemoryCache
from memory.glacier_client import upload_object, download_object

# Set up log directories and files
log_dir = "/root/projects/t1-brain/logs/"
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, "session_memory.log")
token_log_file = os.path.join(log_dir, "token_usage.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file, mode='a'), logging.StreamHandler()]
)
session_logger = logging.getLogger(__name__)
token_logger = logging.getLogger("token_logger")
token_handler = logging.FileHandler(token_log_file, mode='a')
token_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
token_logger.addHandler(token_handler)
token_logger.setLevel(logging.INFO)

# Encoder for tokenization
encoder = tiktoken.encoding_for_model("text-embedding-ada-002")
PROMOTE_THRESHOLD = 5
TIME_THRESHOLD = 86400  # 1 day

def sha256_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()

def chunk_by_tokens(text, max_tokens=300):
    """Chunk text into smaller pieces limited by max_tokens per chunk."""
    words = text.split()
    chunks = []
    current = []
    for word in words:
        current.append(word)
        if len(encoder.encode(" ".join(current))) >= max_tokens:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks

def log_memory_access(session_id, query, matched, match_score=None, memory_type=None):
    """
    Log memory access to the PostgreSQL memory_access_log table.
    
    Args:
        session_id (str): The session identifier.
        query (str): The query text.
        matched (bool): Whether a match was found.
        match_score (float, optional): The match score if applicable.
        memory_type (str, optional): The type of memory accessed.
        
    Returns:
        bool: True if logging was successful, False otherwise.
    """
    if not session_id:
        session_logger.error("❌ [LOGGING] Cannot log memory access: session_id is required")
        return False
    if not query:
        session_logger.error("❌ [LOGGING] Cannot log memory access: query is required")
        return False

    query_truncated = query[:500] if len(query) > 500 else query

    log_conn = None
    log_cursor = None
    try:
        session_logger.debug(f"Attempting to log memory access: session_id={session_id}, query={query_truncated[:30]}..., matched={matched}")
        
        log_conn = psycopg2.connect(
            host=settings.PG_HOST,
            database=settings.PG_DATABASE,
            user=settings.PG_USER,
            password=settings.PG_PASSWORD
        )
        log_cursor = log_conn.cursor()

        log_cursor.execute(
            "INSERT INTO memory_access_log (session_id, query, matched, match_score, memory_type) VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (session_id, query_truncated, matched, match_score, memory_type or "unknown")
        )
        inserted_id = log_cursor.fetchone()[0]
        log_conn.commit()

        session_logger.debug(f"✅ Successfully logged memory access for session_id={session_id}, log_id={inserted_id}")
        return True

    except Exception as e:
        session_logger.error(f"❌ [LOGGING] Failed to log memory access: {type(e).__name__}: {e}")
        session_logger.error(traceback.format_exc())
        if log_conn:
            try:
                log_conn.rollback()
            except Exception as rollback_error:
                session_logger.error(f"❌ [LOGGING] Failed to rollback transaction: {rollback_error}")
        return False

    finally:
        if log_cursor:
            try:
                log_cursor.close()
            except Exception as cursor_error:
                session_logger.error(f"❌ [LOGGING] Failed to close cursor: {cursor_error}")
        if log_conn:
            try:
                log_conn.close()
            except Exception as conn_error:
                session_logger.error(f"❌ [LOGGING] Failed to close connection: {conn_error}")

class PersistentSessionMemory:
    def __init__(self):
        self.ttl = 86400
        self.pg_conn = None
        self.pg_cursor = None
        self.redis_client = redis.StrictRedis(host="localhost", port=6379, decode_responses=True)
        self._warm_cache = WarmMemoryCache.get_instance()
        self._graph_memory = None

    @property
    def graph_memory(self):
        if self._graph_memory is None:
            from memory.graph_memory import GraphMemory
            self._graph_memory = GraphMemory()
        return self._graph_memory

    def connect_to_db(self):
        if not self.pg_conn or self.pg_conn.closed:
            self.pg_conn = psycopg2.connect(
                host=settings.PG_HOST,
                database=settings.PG_DATABASE,
                user=settings.PG_USER,
                password=settings.PG_PASSWORD
            )
            self.pg_cursor = self.pg_conn.cursor()

    def generate_embedding(self, content):
        try:
            api_key = get_api_key("text")
            client = OpenAI(api_key=api_key)
            response = client.embeddings.create(model="text-embedding-ada-002", input=content)
            usage = getattr(response, 'usage', None)
            if usage:
                token_logger.info(f"embedding | tokens={usage.total_tokens} | model=text-embedding-ada-002")
            return response.data[0].embedding
        except Exception as e:
            session_logger.error(f"❌ Embedding error: {e}")
            return np.zeros(1536).tolist()

    def should_promote(self, memory):
        access_score = memory.get("access_score", 0)
        created_at = memory.get("timestamp", time.time())
        return access_score >= PROMOTE_THRESHOLD or (time.time() - created_at > TIME_THRESHOLD)

    def update_postgres(self, session_id, memory, sha, embedding=None):
        try:
            self.connect_to_db()
            embedding_str = "[" + ",".join(map(str, embedding)) + "]" if embedding else None
            access_score = memory.get("access_score", 0)
            last_accessed = memory.get("last_accessed", time.time())

            self.pg_cursor.execute("SELECT 1 FROM embeddings WHERE sha_hash = %s", (sha,))
            record_exists = self.pg_cursor.fetchone() is not None

            if record_exists:
                sql = """UPDATE embeddings SET access_score = %s, last_accessed = to_timestamp(%s)
                         WHERE sha_hash = %s RETURNING id, access_score"""
                params = (access_score, last_accessed, sha)
            else:
                sql = """INSERT INTO embeddings (
                            session_id, query, response, embedding, memory_type, sentiment,
                            source_type, image_url, sha_hash, access_score, last_accessed, created_at
                         ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, to_timestamp(%s), NOW())
                         RETURNING id, access_score"""
                params = (
                    session_id, memory["query"], memory["response"], embedding_str,
                    memory.get("memory_type", "semantic"), memory.get("sentiment", "neutral"),
                    memory.get("source_type", "text"), memory.get("image_url", None),
                    sha, access_score, last_accessed
                )

            self.pg_cursor.execute(sql, params)
            result = self.pg_cursor.fetchone()
            self.pg_conn.commit()

            if result:
                pg_id, pg_score = result
                session_logger.info(f"✅ PostgreSQL updated: id={pg_id}, sha={sha[:10]}, score={pg_score}")
                if embedding and hasattr(self, '_warm_cache'):
                    self._warm_cache.add(session_id, memory["query"], memory["response"],
                                         embedding, access_score=access_score, last_accessed=last_accessed)
                return True
            return False
        except Exception as e:
            if self.pg_conn:
                self.pg_conn.rollback()
            session_logger.error(f"❌ PostgreSQL update error for sha={sha[:10]}: {e}")
            return False

    def retrieve_memory(self, session_id, query):
        try:
            self.connect_to_db()
            key = f"memory:{session_id}"

            if self.redis_client.exists(key):
                memory_items = [json.loads(m) for m in self.redis_client.lrange(key, 0, -1)]
                updated = []
                for m in memory_items:
                    m["access_score"] = m.get("access_score", 0) + 1
                    m["last_accessed"] = time.time()
                    sha = sha256_hash(m["query"] + m["response"])
                    
                    log_success = log_memory_access(session_id, query, True, m.get("access_score", 1), m.get("memory_type") or "redis")
                    if not log_success:
                        session_logger.warning(f"Failed to log memory access for session_id={session_id}")
                    
                    if self.should_promote(m):
                        embedding = self.generate_embedding(m["query"])
                        self.update_postgres(session_id, m, sha, embedding)
                    else:
                        self.update_postgres(session_id, m, sha)
                    updated.append(json.dumps(m))
                pipe = self.redis_client.pipeline()
                pipe.delete(key)
                pipe.rpush(key, *updated)
                pipe.expire(key, self.ttl)
                pipe.execute()
                return [json.loads(m) for m in updated]

            embedding = self.generate_embedding(query)
            warm_hits = self._warm_cache.find_similar(embedding)
            if warm_hits:
                for hit in warm_hits:
                    hit["access_score"] = hit.get("access_score", 0) + 1
                    hit["last_accessed"] = time.time()
                    sha = sha256_hash(hit["query"] + hit["response"])
                    self.update_postgres(session_id, hit, sha)
                    
                    log_success = log_memory_access(session_id, query, True, hit.get("access_score"), hit.get("memory_type", "faiss"))
                    if not log_success:
                        session_logger.warning(f"Failed to log memory access for session_id={session_id}")
                        
                return warm_hits

            glacier_key = f"{session_id}_glacier.json"
            glacier_blob = download_object(glacier_key)
            if glacier_blob:
                glacier_data = json.loads(glacier_blob) if isinstance(glacier_blob, str) else glacier_blob
                if isinstance(glacier_data, dict):
                    glacier_data = [glacier_data]
                for m in glacier_data:
                    m["access_score"] = m.get("access_score", 0) + 1
                    m["last_accessed"] = time.time()
                    self.redis_client.rpush(key, json.dumps(m))
                    
                    log_success = log_memory_access(session_id, query, True, m.get("access_score"), m.get("memory_type", "glacier"))
                    if not log_success:
                        session_logger.warning(f"Failed to log memory access for session_id={session_id}")
                        
                self.redis_client.expire(key, self.ttl)
                return glacier_data

            log_success = log_memory_access(session_id, query, False)
            if not log_success:
                session_logger.warning(f"Failed to log memory access for session_id={session_id}")
            return []

        except Exception as e:
            session_logger.error(f"❌ Retrieval error: {e}")
            session_logger.error(traceback.format_exc())
            return []

    def enrich_to_graph(self, session_id, query, memory_data):
        """
        Enrich memory with graph data by linking memory chunks to intent, topic, and emotion.
        
        Args:
            session_id (str): The session identifier.
            query (str): The query text.
            memory_data (dict): The memory dictionary containing metadata.
            
        Returns:
            bool: True if enrichment was successful, False otherwise.
        """
        try:
            # Extract enrichment metadata from memory_data
            intent = memory_data.get("intent")
            topic = memory_data.get("topic")
            emotion = memory_data.get("sentiment") or memory_data.get("emotion")
            user_id = memory_data.get("user_id")
            
            session_logger.info(f"🚀 Enriching Query: {query[:50]} | intent={intent}, topic={topic}, emotion={emotion}")
            
            entities = memory_data.get("entities", [])
            keywords = memory_data.get("keywords", [])
            memory_type = memory_data.get("memory_type", "semantic")
            response = memory_data.get("response", "")
            
            context_props = {
                "session_id": session_id,
                "memory_type": memory_type,
                "timestamp": memory_data.get("timestamp", time.time()),
            }
            if user_id:
                context_props["user_id"] = user_id
                
            if any([intent, topic, emotion, entities, keywords]):
                enrichment_result = self.graph_memory.add_context_nodes(
                    query,
                    response=response,
                    intent=intent,
                    topic=topic,
                    emotion=emotion,
                    entities=entities,
                    context_props=context_props
                )
                if keywords and isinstance(keywords, list):
                    for keyword in keywords:
                        if keyword and isinstance(keyword, str):
                            try:
                                self.graph_memory.add_keyword_relationship(query, keyword)
                            except Exception as kw_error:
                                session_logger.debug(f"⚠️ Error adding keyword relationship for '{keyword}': {kw_error}")
                
                session_logger.info(f"✅ Neo4j enrichment successful for session={session_id}, query={query[:30]}...")
                return True
            else:
                session_logger.warning(f"⚠️ No enrichment metadata available for graph storage, session={session_id}")
                return False
                
        except Exception as e:
            session_logger.error(f"❌ Neo4j enrichment failed: {e}")
            session_logger.error(traceback.format_exc())
            return False

    def store_memory(self, session_id, query, response, memory_type="semantic", sentiment="neutral", source_type="text", metadata=None):
        """
        Stores a memory entry into Redis, promotes to PostgreSQL/FAISS as needed,
        uploads to Glacier for long-term backup, and enriches the memory into the graph.
        
        Graph enrichment is performed BEFORE response chunking to ensure full context.
        """
        try:
            if not query or not response:
                session_logger.warning("❌ Missing query or response, skipping memory storage")
                return False

            if not metadata:
                metadata = {}

            timestamp = time.time()
            full_memory = {
                "query": query,
                "response": response,  # Full response for graph enrichment
                "memory_type": memory_type,
                "sentiment": sentiment,
                "source_type": source_type,
                "timestamp": timestamp,
                "access_score": 1,
                "last_accessed": timestamp
            }
            # Incorporate additional metadata
            for k, v in metadata.items():
                full_memory[k] = v

            # Perform Neo4j graph enrichment BEFORE chunking the response
            self.enrich_to_graph(session_id, query, full_memory)
            
            # Proceed with token-based chunking of the response for Redis storage
            chunks = chunk_by_tokens(response, max_tokens=300)
            full_memory["total_chunks"] = len(chunks)
            key = f"memory:{session_id}"

            for i, chunk in enumerate(chunks):
                chunked_memory = full_memory.copy()
                chunked_memory["response"] = chunk
                chunked_memory["chunk_id"] = i

                memory_json = json.dumps(chunked_memory)
                self.redis_client.rpush(key, memory_json)

                if self.should_promote(chunked_memory):
                    sha = sha256_hash(query + chunk)
                    embedding = self.generate_embedding(query)
                    self.update_postgres(session_id, chunked_memory, sha, embedding)
            
            self.redis_client.expire(key, self.ttl)
            
            # Backup to Glacier if multiple memory items exist
            if len(chunks) > 1:
                memory_items = [json.loads(m) for m in self.redis_client.lrange(key, 0, -1)]
                if memory_items:
                    glacier_key = f"{session_id}_glacier.json"
                    glacier_data = json.dumps(memory_items)
                    upload_object(glacier_key, glacier_data)
                    session_logger.info(f"✅ Uploaded {len(memory_items)} memories to glacier: {glacier_key}")

            return True

        except Exception as e:
            session_logger.error(f"❌ Memory storage error: {e}")
            session_logger.error(traceback.format_exc())
            return False

    def test_memory_access_logging(self):
        """
        Test function to verify memory access logging functionality.
        """
        test_session_id = f"test_session_{int(time.time())}"
        test_query = "Test query for logging verification"
        test_matched = True
        test_score = 0.95
        test_memory_type = "test"
        
        session_logger.info(f"Testing memory access logging with session_id: {test_session_id}")
        
        success = log_memory_access(
            test_session_id, 
            test_query, 
            test_matched, 
            test_score, 
            test_memory_type
        )
        
        if success:
            session_logger.info(f"✅ Test log entry created successfully for session {test_session_id}")
        else:
            session_logger.error(f"❌ Failed to create test log entry for session {test_session_id}")
        
        conn = None
        cursor = None
        try:
            conn = psycopg2.connect(
                host=settings.PG_HOST,
                database=settings.PG_DATABASE,
                user=settings.PG_USER,
                password=settings.PG_PASSWORD
            )
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, session_id, query, matched, match_score, memory_type, created_at FROM memory_access_log WHERE session_id = %s",
                (test_session_id,)
            )
            rows = cursor.fetchall()
            if rows:
                session_logger.info(f"✅ Found {len(rows)} log entries in database for session {test_session_id}")
                for row in rows:
                    session_logger.info(f"  - ID: {row[0]}, Session: {row[1]}, Query: {row[2][:30]}..., Matched: {row[3]}, Score: {row[4]}, Type: {row[5]}, Created: {row[6]}")
                return True
            else:
                session_logger.error(f"❌ No log entries found in database for session {test_session_id}")
                return False
        except Exception as e:
            session_logger.error(f"❌ Error verifying log entry: {type(e).__name__}: {e}")
            session_logger.error(traceback.format_exc())
            return False
        finally:
            if cursor:
                try:
                    cursor.close()
                except:
                    pass
            if conn:
                try:
                    conn.close()
                except:
                    pass

    def test_graph_enrichment(self, session_id=None):
        """
        Test function to verify graph enrichment functionality.
        
        Args:
            session_id (str, optional): Session ID for testing; if None, a test ID is generated.
            
        Returns:
            bool: True if the test passed successfully, False otherwise.
        """
        if not session_id:
            session_id = f"test_graph_{int(time.time())}"
            
        test_query = "How do I improve my machine learning model?"
        test_response = "You can improve your model by collecting more data, feature engineering, and fine-tuning hyperparameters."
        test_metadata = {
            "intent": "learn",
            "topic": "machine learning",
            "sentiment": "curious",
            "entities": ["machine learning model", "data"],
            "keywords": ["improvement", "hyperparameters", "feature engineering"]
        }
        
        session_logger.info(f"Testing graph enrichment with session_id: {session_id}")
        
        try:
            store_success = self.store_memory(
                session_id,
                test_query,
                test_response,
                memory_type="semantic",
                sentiment="curious",
                metadata=test_metadata
            )
            
            if store_success:
                session_logger.info(f"✅ Test memory stored with graph enrichment: {session_id}")
                if self.graph_memory.find_related_contexts(test_query):
                    session_logger.info(f"✅ Graph connections verified for query: {test_query[:30]}...")
                    return True
                else:
                    session_logger.warning("⚠️ No graph connections found for test query")
                    return False
            else:
                session_logger.error("❌ Failed to store test memory with graph enrichment")
                return False
        except Exception as e:
            session_logger.error(f"❌ Graph enrichment test error: {type(e).__name__}: {e}")
            session_logger.error(traceback.format_exc())
            return False
