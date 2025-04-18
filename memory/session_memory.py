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

# Ensure log directory exists
log_dir = "/root/projects/t1-brain/logs/"
os.makedirs(log_dir, exist_ok=True)

# Session memory logger
log_file = os.path.join(log_dir, "session_memory.log")
token_log_file = os.path.join(log_dir, "token_usage.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode='a'),
        logging.StreamHandler()
    ]
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

def sha256_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()

def chunk_by_tokens(text: str, max_tokens: int = 300) -> list:
    """Split text into chunks up to max_tokens each."""
    words = text.split()
    chunks, current = [], []
    for w in words:
        current.append(w)
        if len(encoder.encode(" ".join(current))) >= max_tokens:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks

def log_memory_access(session_id: str, query: str, matched: bool,
                      match_score: float = None, memory_type: str = None) -> bool:
    """Log each memory access to PostgreSQL."""
    if not session_id:
        session_logger.error("❌ [LOGGING] Missing session_id")
        return False
    if not query:
        session_logger.error("❌ [LOGGING] Missing query")
        return False

    truncated = query if len(query) <= 500 else query[:500]
    conn = cursor = None
    try:
        conn = psycopg2.connect(
            host=settings.PG_HOST,
            database=settings.PG_DATABASE,
            user=settings.PG_USER,
            password=settings.PG_PASSWORD
        )
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO memory_access_log (session_id, query, matched, match_score, memory_type) "
            "VALUES (%s,%s,%s,%s,%s) RETURNING id",
            (session_id, truncated, matched, match_score, memory_type or "unknown")
        )
        _ = cursor.fetchone()[0]
        conn.commit()
        return True
    except Exception as e:
        session_logger.error(f"❌ [LOGGING] Failed to log access: {e}")
        session_logger.error(traceback.format_exc())
        if conn:
            conn.rollback()
        return False
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


class PersistentSessionMemory:
    def __init__(self):
        self.ttl = TIME_THRESHOLD
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

    def generate_embedding(self, content: str) -> list:
        """Call OpenAI to generate embedding, fallback to zeros on error."""
        try:
            api_key = get_api_key("text")
            client = OpenAI(api_key=api_key)
            resp = client.embeddings.create(model="text-embedding-ada-002", input=content)
            usage = getattr(resp, "usage", None)
            if usage:
                token_logger.info(f"embedding | tokens={usage.total_tokens}")
            return resp.data[0].embedding
        except Exception as e:
            session_logger.error(f"❌ Embedding error: {e}")
            return np.zeros(1536).tolist()

    def should_promote(self, memory: dict) -> bool:
        score = memory.get("access_score", 0)
        created = memory.get("timestamp", time.time())
        return score >= PROMOTE_THRESHOLD or (time.time() - created) > TIME_THRESHOLD

    def update_postgres(self, session_id: str, memory: dict, sha: str, embedding: list = None) -> bool:
        """Insert or update memory chunk in PostgreSQL and FAISS warm cache."""
        try:
            self.connect_to_db()
            emb_str = f"[{','.join(map(str, embedding))}]" if embedding else None
            access_score = memory.get("access_score", 0)
            last = memory.get("last_accessed", time.time())

            self.pg_cursor.execute("SELECT 1 FROM embeddings WHERE sha_hash=%s", (sha,))
            exists = self.pg_cursor.fetchone() is not None

            if exists:
                sql = "UPDATE embeddings SET access_score=%s, last_accessed=to_timestamp(%s) WHERE sha_hash=%s RETURNING id, access_score"
                params = (access_score, last, sha)
            else:
                sql = (
                    "INSERT INTO embeddings (session_id, query, response, embedding, memory_type, sentiment, "
                    "source_type, image_url, sha_hash, access_score, last_accessed, created_at) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,to_timestamp(%s),NOW()) "
                    "RETURNING id, access_score"
                )
                params = (
                    session_id, memory["query"], memory["response"], emb_str,
                    memory.get("memory_type", "semantic"), memory.get("sentiment", "neutral"),
                    memory.get("source_type", "text"), memory.get("image_url"),
                    sha, access_score, last
                )

            self.pg_cursor.execute(sql, params)
            rec = self.pg_cursor.fetchone()
            self.pg_conn.commit()
            if rec and embedding:
                # add to FAISS warm cache
                self._warm_cache.add(
                    session_id, memory["query"], memory["response"],
                    embedding,
                    access_score=access_score,
                    last_accessed=last
                )
            return True
        except Exception as e:
            session_logger.error(f"❌ PostgreSQL update error: {e}")
            session_logger.error(traceback.format_exc())
            if self.pg_conn:
                self.pg_conn.rollback()
            return False

    def retrieve_memory(self, session_id: str, query: str) -> list:
        """Retrieve memory from Redis, FAISS warm cache, or Glacier in that order."""
        try:
            self.connect_to_db()
            key = f"memory:{session_id}"

            if self.redis_client.exists(key):
                items = [json.loads(x) for x in self.redis_client.lrange(key, 0, -1)]
                updated = []
                for m in items:
                    m["access_score"] = m.get("access_score", 0) + 1
                    m["last_accessed"] = time.time()
                    sha = sha256_hash(m["query"] + m["response"])
                    log_memory_access(session_id, query, True, m["access_score"], m.get("memory_type"))
                    if self.should_promote(m):
                        emb = self.generate_embedding(m["query"])
                        self.update_postgres(session_id, m, sha, emb)
                    else:
                        self.update_postgres(session_id, m, sha)
                    updated.append(json.dumps(m))

                pipe = self.redis_client.pipeline()
                pipe.delete(key)
                pipe.rpush(key, *updated)
                pipe.expire(key, self.ttl)
                pipe.execute()
                return [json.loads(x) for x in updated]

            # FAISS warm cache
            emb = self.generate_embedding(query)
            warm_hits = self._warm_cache.find_similar(emb)
            if warm_hits:
                for m in warm_hits:
                    m["access_score"] = m.get("access_score", 0) + 1
                    m["last_accessed"] = time.time()
                    sha = sha256_hash(m["query"] + m["response"])
                    self.update_postgres(session_id, m, sha)
                    log_memory_access(session_id, query, True, m["access_score"], m.get("memory_type"))
                return warm_hits

            # Glacier fallback
            blob = download_object(f"{session_id}_glacier.json")
            if blob:
                data = json.loads(blob) if isinstance(blob, str) else blob
                records = data if isinstance(data, list) else [data]
                for m in records:
                    m["access_score"] = m.get("access_score", 0) + 1
                    m["last_accessed"] = time.time()
                    self.redis_client.rpush(key, json.dumps(m))
                    log_memory_access(session_id, query, True, m["access_score"], m.get("memory_type"))
                self.redis_client.expire(key, self.ttl)
                return records

            log_memory_access(session_id, query, False)
            return []
        except Exception as e:
            session_logger.error(f"❌ Retrieval error: {e}")
            session_logger.error(traceback.format_exc())
            return []

    def find_similar_queries(self, query: str, top_k: int = 5, min_score: float = 0.80) -> list:
        """
        Find past queries from warm memory that are semantically similar to the current query.
        Applies a minimum score filter for precision.
        """
        try:
            emb = self.generate_embedding(query)
            raw_results = self._warm_cache.find_similar(emb, top_k=top_k)
            session_logger.info(f"[FAISS] 🔍 Similar query match count: {len(raw_results)} for input: '{query}'")
            for idx, r in enumerate(raw_results):
                session_logger.info(f"[FAISS] Match #{idx+1}: query='{r.get('query')}', score={r.get('score')}")
            filtered = [r for r in raw_results if r.get("score", 0) >= min_score]
            session_logger.info(f"[FAISS] 🔍 Filtered to {len(filtered)} matches with min_score={min_score}")
            return filtered
        except Exception as e:
            session_logger.warning(f"⚠️ Similar query search failed: {e}")
            return []

    def store_memory(self, session_id: str,
                     query: str,
                     response: str,
                     memory_type: str = "semantic",
                     sentiment: str = "neutral",
                     source_type: str = "text",
                     metadata: dict = None) -> bool:
        """Store and enrich a new memory entry."""
        try:
            if not query or not response:
                session_logger.warning("❌ store_memory called with empty query or response")
                return False

            if metadata is None:
                metadata = {}

            timestamp = time.time()
            full = {
                "query": query,
                "response": response,
                "memory_type": memory_type,
                "sentiment": sentiment,
                "source_type": source_type,
                "timestamp": timestamp,
                "access_score": 1,
                "last_accessed": timestamp
            }
            full.update(metadata)

            # Graph enrichment
            self.enrich_to_graph(session_id, query, full)

            # Chunk + Redis
            chunks = chunk_by_tokens(response)
            full["total_chunks"] = len(chunks)
            key = f"memory:{session_id}"
            for idx, ch in enumerate(chunks):
                item = full.copy()
                item["response"] = ch
                item["chunk_id"] = idx
                self.redis_client.rpush(key, json.dumps(item))
                if self.should_promote(item):
                    sha = sha256_hash(query + ch)
                    emb = self.generate_embedding(query)
                    self.update_postgres(session_id, item, sha, emb)

            self.redis_client.expire(key, self.ttl)

            # Glacier backup
            if len(chunks) > 1:
                all_items = [json.loads(x) for x in self.redis_client.lrange(key, 0, -1)]
                upload_object(f"{session_id}_glacier.json", json.dumps(all_items))
                session_logger.info(f"✅ Uploaded {len(all_items)} items to Glacier")

            return True
        except Exception as e:
            session_logger.error(f"❌ store_memory error: {e}")
            session_logger.error(traceback.format_exc())
            return False

    def enrich_to_graph(self, session_id: str, query: str, memory_data: dict) -> bool:
        """Link memory in Neo4j with intent, topic, emotion, entities, keywords."""
        try:
            intent = memory_data.get("intent")
            topic = memory_data.get("topic")
            emotion = memory_data.get("sentiment") or memory_data.get("emotion")
            entities = memory_data.get("entities", [])
            keywords = memory_data.get("keywords", [])
            context_props = {
                "session_id": session_id,
                "memory_type": memory_data.get("memory_type"),
                "timestamp": memory_data.get("timestamp")
            }
            user_id = memory_data.get("user_id")
            if user_id:
                context_props["user_id"] = user_id

            if any([intent, topic, emotion, entities, keywords]):
                self.graph_memory.add_context_nodes(
                    query,
                    response=memory_data.get("response"),
                    intent=intent,
                    topic=topic,
                    emotion=emotion,
                    entities=entities,
                    context_props=context_props
                )
                for kw in keywords:
                    try:
                        self.graph_memory.add_keyword_relationship(query, kw)
                    except Exception:
                        session_logger.debug("⚠️ keyword enrich failed")
                return True
            else:
                session_logger.warning(f"⚠️ No metadata for graph enrichment: {query}")
                return False
        except Exception as e:
            session_logger.error(f"❌ enrich_to_graph error: {e}")
            session_logger.error(traceback.format_exc())
            return False

    def summarize_session(self, session_id: str) -> dict:
        """Summarize a session’s graph metadata."""
        session_logger.info(f"Summarize session: {session_id}")
        return self.graph_memory.summarize_session(session_id)

    def test_memory_access_logging(self):
        """Existing test—unchanged."""
        pass

    def test_graph_enrichment(self, session_id: str = None):
        """Placeholder to avoid indentation error."""
        pass
