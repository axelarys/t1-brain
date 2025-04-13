# session_memory.py

import os, json, time, redis, logging, psycopg2, hashlib, numpy as np, tiktoken
from datetime import datetime
from config import settings
from openai import OpenAI
from utils.memory_utils import get_api_key
from memory.warm_layer import WarmMemoryCache
from memory.glacier_client import upload_object, download_object

log_dir = "/root/projects/t1-brain/logs/"
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, "session_memory.log")
token_log_file = os.path.join(log_dir, "token_usage.log")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file, mode='a'), logging.StreamHandler()])
session_logger = logging.getLogger(__name__)

token_logger = logging.getLogger("token_logger")
token_handler = logging.FileHandler(token_log_file, mode='a')
token_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
token_logger.addHandler(token_handler)
token_logger.setLevel(logging.INFO)

encoder = tiktoken.encoding_for_model("text-embedding-ada-002")

# Constants
PROMOTE_THRESHOLD = 5  # Promotion access score threshold
TIME_THRESHOLD = 24 * 60 * 60  # 24 hours in seconds for time-based promotion

def chunk_by_tokens(text, max_tokens=300):
    words, chunks, current = text.split(), [], []
    for word in words:
        current.append(word)
        if len(encoder.encode(" ".join(current))) >= max_tokens:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks

def sha256_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()

class PersistentSessionMemory:
    def __init__(self):
        self.ttl = 86400
        self.pg_conn = None
        self.pg_cursor = None
        self.redis_client = redis.StrictRedis(host="localhost", port=6379, decode_responses=True)
        self._graph_memory = None
        self._warm_cache = WarmMemoryCache.get_instance()

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

    def cleanup_expired_memory(self, session_id):
        try:
            key = f"memory:{session_id}"
            if not self.redis_client.exists(key):
                return {"status": "skipped", "message": "No memory to clean."}
            now = time.time()
            valid_memory, removed_count = [], 0
            all_memory = self.redis_client.lrange(key, 0, -1)
            self.redis_client.delete(key)
            for m in all_memory:
                try:
                    m_obj = json.loads(m)
                    policy = m_obj.get("memory_policy", {})
                    expiration = policy.get("expiration", 0)
                    short_term = policy.get("short_term", False)
                    timestamp = m_obj.get("timestamp", 0)
                    if short_term and expiration and now > (timestamp + expiration):
                        removed_count += 1
                        continue
                    valid_memory.append(json.dumps(m_obj))
                except Exception as e:
                    session_logger.warning(f"⚠️ Failed parsing memory item: {e}")
            for v in valid_memory:
                self.redis_client.rpush(key, v)
            self.redis_client.expire(key, self.ttl)
            return {"status": "cleaned", "removed": removed_count, "retained": len(valid_memory)}
        except Exception as e:
            session_logger.error(f"❌ Cleanup error: {e}")
            return {"status": "error", "message": "Cleanup failed"}

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

    def store_memory(self, session_id, query, response, memory_type="semantic", sentiment="neutral", source_type="text", metadata=None):
        self.connect_to_db()
        try:
            session_key = f"session:{session_id}"
            if not self.redis_client.exists(session_key):
                self.redis_client.setex(session_key, self.ttl, json.dumps({"session_id": session_id}))
                session_logger.info(f"✅ New session key: {session_id}")
            self.cleanup_expired_memory(session_id)

            is_image = memory_type == "image"
            image_url = None
            if is_image:
                try:
                    api_key = get_api_key("image")
                    client = OpenAI(api_key=api_key)
                    extraction = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe this image for memory embedding."},
                                {"type": "image_url", "image_url": {"url": query}}
                            ]
                        }]
                    )
                    description = extraction.choices[0].message.content.strip()
                    session_logger.info(f"🧠 Extracted image description: {description}")
                    image_url = query
                    query = description
                except Exception as e:
                    session_logger.error(f"❌ Image analysis failed: {e}")
                    return {"status": "error", "message": "Image analysis failed"}

            chunks = [query] if is_image else chunk_by_tokens(query) if len(encoder.encode(query)) > 400 else [query]
            for i, chunk in enumerate(chunks):
                chunk_id = f"{session_id}_chunk_{i+1}"
                dedup_hash = sha256_hash(chunk + response)
                self.pg_cursor.execute("SELECT 1 FROM embeddings WHERE sha_hash = %s LIMIT 1", (dedup_hash,))
                if self.pg_cursor.fetchone():
                    session_logger.info(f"⚠️ Duplicate memory skipped (hash: {dedup_hash})")
                    continue

                embedding = self.generate_embedding(chunk)
                if not embedding or len(embedding) != 1536:
                    session_logger.warning("⚠️ Skipping FAISS insert: invalid embedding size")
                else:
                    current_time = time.time()
                    self._warm_cache.add(
                        session_id, 
                        chunk, 
                        response, 
                        embedding, 
                        access_score=0, 
                        last_accessed=current_time
                    )
                    session_logger.info(f"🔥 FAISS Add: session={session_id}, chunk={chunk[:50]}..., index_total={self._warm_cache.index.ntotal}")

                embedding_str = "[" + ",".join(map(str, embedding)) + "]"
                current_time = time.time()
                memory_metadata = {
                    "query": chunk,
                    "response": response,
                    "timestamp": current_time,
                    "memory_type": memory_type,
                    "sentiment": sentiment,
                    "chunk_id": chunk_id,
                    "source_type": "image" if is_image else source_type,
                    "image_url": image_url if is_image else None,
                    "access_score": 0,
                    "last_accessed": current_time,
                    "memory_policy": {
                        "short_term": memory_type != "summary",
                        "expiration": 3 * 86400 if memory_type != "summary" else 0
                    }
                }

                if metadata:
                    memory_metadata.update(metadata)

                self.redis_client.rpush(f"memory:{session_id}", json.dumps(memory_metadata))
                self.redis_client.expire(f"memory:{session_id}", self.ttl)

                # Include access_score and last_accessed in PostgreSQL insertion
                self.pg_cursor.execute(
                    """INSERT INTO embeddings (
                        session_id, query, response, embedding, memory_type, sentiment, 
                        source_type, image_url, sha_hash, created_at, access_score, last_accessed
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), %s, NOW())""",
                    (
                        session_id,
                        chunk,
                        response,
                        embedding_str,
                        memory_type,
                        sentiment,
                        memory_metadata["source_type"],
                        memory_metadata["image_url"],
                        dedup_hash,
                        memory_metadata["access_score"]
                    )
                )
                self.pg_conn.commit()
                self.graph_memory.store_graph_memory(session_id, chunk, response, memory_type, sentiment)

                glacier_key = f"{session_id}_{int(current_time)}.json"
                success = upload_object(glacier_key, memory_metadata)
                if success:
                    session_logger.info(f"✅ Glacier upload succeeded: {glacier_key}")
                else:
                    session_logger.warning(f"⚠️ Glacier upload failed: {glacier_key}")

        except Exception as e:
            session_logger.error(f"❌ Store error: {e}")
            return {"status": "error", "message": "Memory store failed"}
        return {"status": "stored", "message": f"Memory stored in {len(chunks)} chunk(s)."}

    def should_promote(self, memory_item):
        """Determine if memory should be promoted based on combined criteria."""
        access_score = memory_item.get("access_score", 0)
        created_time = memory_item.get("timestamp", time.time())
        current_time = time.time()
        
        # Promote if access score threshold met OR time threshold passed
        return (access_score >= PROMOTE_THRESHOLD or 
                (current_time - created_time) > TIME_THRESHOLD)

    def update_postgres_memory(self, session_id, memory_item, sha_hash, embedding=None):
        """Update or insert memory in PostgreSQL with proper error handling."""
        try:
            self.connect_to_db()
            # Check if entry exists
            self.pg_cursor.execute("SELECT 1 FROM embeddings WHERE sha_hash = %s LIMIT 1", (sha_hash,))
            exists = self.pg_cursor.fetchone()
            
            if exists:
                # Update existing record
                self.pg_cursor.execute(
                    """UPDATE embeddings 
                       SET access_score = %s, 
                           last_accessed = NOW() 
                       WHERE sha_hash = %s""",
                    (memory_item["access_score"], sha_hash)
                )
                self.pg_conn.commit()
                return True
            elif embedding and len(embedding) == 1536:
                # Insert new record with full metadata
                embedding_str = "[" + ",".join(map(str, embedding)) + "]"
                self.pg_cursor.execute(
                    """INSERT INTO embeddings (
                        session_id, query, response, embedding, memory_type, sentiment,
                        source_type, image_url, sha_hash, created_at, access_score, last_accessed
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), %s, NOW())""",
                    (
                        session_id,
                        memory_item["query"],
                        memory_item["response"],
                        embedding_str,
                        memory_item.get("memory_type", "semantic"),
                        memory_item.get("sentiment", "neutral"),
                        memory_item.get("source_type", "text"),
                        memory_item.get("image_url", None),
                        sha_hash,
                        memory_item["access_score"]
                    )
                )
                self.pg_conn.commit()
                # Also update FAISS
                self._warm_cache.add(
                    session_id, 
                    memory_item["query"], 
                    memory_item["response"], 
                    embedding,
                    access_score=memory_item["access_score"],
                    last_accessed=memory_item["last_accessed"]
                )
                return True
            return False
        except Exception as e:
            self.pg_conn.rollback()
            session_logger.error(f"❌ PostgreSQL update failed: {e}")
            return False

    def batch_process_promotions(self, session_id, promotion_candidates):
        """Process multiple promotions in an efficient batch."""
        success_count = 0
        for memory in promotion_candidates:
            embedding = self.generate_embedding(memory["query"])
            sha_hash = sha256_hash(memory["query"] + memory["response"])
            if self.update_postgres_memory(session_id, memory, sha_hash, embedding):
                success_count += 1
                session_logger.info(f"🚀 Promoted to PostgreSQL + FAISS (score={memory['access_score']})")
        
        return success_count

    def retrieve_memory(self, session_id, query):
        try:
            self.connect_to_db()
            key = f"memory:{session_id}"

            if self.redis_client.exists(key):
                session_logger.info(f"🧠 Memory retrieved from Redis for session: {session_id}")
                memories = [json.loads(m) for m in self.redis_client.lrange(key, 0, -1)]

                updated = []
                promotion_candidates = []
                
                # First pass: update access scores
                for m in memories:
                    m["access_score"] = m.get("access_score", 0) + 1
                    m["last_accessed"] = time.time()
                    updated.append(m)
                    
                    # Check if memory should be promoted
                    if self.should_promote(m):
                        promotion_candidates.append(m)
                        
                    # Always update PostgreSQL if record exists
                    sha = sha256_hash(m["query"] + m["response"])
                    self.update_postgres_memory(session_id, m, sha)
                
                # Second pass: batch process promotions
                if promotion_candidates:
                    promoted = self.batch_process_promotions(session_id, promotion_candidates)
                    if promoted:
                        session_logger.info(f"✅ Batch promoted {promoted} memories")
                
                # Update Redis with new memory states
                self.redis_client.delete(key)
                for m in updated:
                    self.redis_client.rpush(key, json.dumps(m))
                self.redis_client.expire(key, self.ttl)

                return updated

            # FAISS fallback
            embedding = self.generate_embedding(query)
            warm_hits = self._warm_cache.find_similar(embedding)
            if warm_hits:
                session_logger.info(f"🔥 FAISS fallback returned {len(warm_hits)} result(s)")
                for hit in warm_hits:
                    session_logger.info(f"➡️ Match | dist={hit.get('distance', 0):.4f} | query={hit['query']}")
                    
                    # Update access stats in PostgreSQL for FAISS hits
                    sha = sha256_hash(hit["query"] + hit["response"])
                    hit["access_score"] = hit.get("access_score", 0) + 1
                    hit["last_accessed"] = time.time()
                    self.update_postgres_memory(session_id, hit, sha)
                
                return warm_hits

            # Glacier fallback
            glacier_key = f"{session_id}_glacier.json"
            glacier_blob = download_object(glacier_key)
            if glacier_blob:
                try:
                    glacier_data = json.loads(glacier_blob) if isinstance(glacier_blob, str) else glacier_blob
                    session_logger.info(f"❄️ Memory restored from Glacier Layer for session: {session_id}")
                    if isinstance(glacier_data, dict):
                        glacier_data = [glacier_data]
                    for m in glacier_data:
                        m["access_score"] = m.get("access_score", 0) + 1
                        m["last_accessed"] = time.time()
                        self.redis_client.rpush(key, json.dumps(m))
                    self.redis_client.expire(key, self.ttl)
                    return glacier_data
                except Exception as e:
                    session_logger.error(f"❌ Glacier decoding error: {e}")
                    return []

            session_logger.warning("⚠️ No memory found in Redis, FAISS, or Glacier.")
            return []

        except Exception as e:
            session_logger.error(f"❌ Retrieval error: {e}")
            return []

    def run_consistency_check(self, session_id=None):
        """
        Run a consistency check between FAISS and PostgreSQL to ensure synchronization.
        Can be scheduled to run periodically.
        """
        try:
            self.connect_to_db()
            
            # Build query for checking embeddings
            query = "SELECT session_id, query, response, embedding, sha_hash, access_score FROM embeddings"
            params = []
            if session_id:
                query += " WHERE session_id = %s"
                params.append(session_id)
            
            self.pg_cursor.execute(query, params)
            records = self.pg_cursor.fetchall()
            
            consistency_issues = 0
            fixed_issues = 0
            
            for record in records:
                rec_session_id, query, response, embedding_str, sha_hash, access_score = record
                
                # Convert string embedding to vector
                try:
                    embedding = json.loads(embedding_str.replace("[", "[").replace("]", "]"))
                    
                    # Check if in FAISS
                    in_faiss = self._warm_cache.exists(rec_session_id, query, response)
                    if not in_faiss:
                        consistency_issues += 1
                        # Add to FAISS
                        self._warm_cache.add(
                            rec_session_id, 
                            query, 
                            response, 
                            embedding,
                            access_score=access_score,
                            last_accessed=time.time()
                        )
                        fixed_issues += 1
                        session_logger.info(f"🔄 Fixed consistency: Added missing FAISS entry for {sha_hash}")
                except Exception as e:
                    session_logger.error(f"❌ Consistency check error for {sha_hash}: {e}")
            
            if consistency_issues > 0:
                session_logger.info(f"🔍 Consistency check: Found {consistency_issues} issues, fixed {fixed_issues}")
            else:
                session_logger.info(f"✅ Consistency check: No issues found")
                
            return {"status": "completed", "issues_found": consistency_issues, "issues_fixed": fixed_issues}
        except Exception as e:
            session_logger.error(f"❌ Consistency check failed: {e}")
            return {"status": "error", "message": str(e)}