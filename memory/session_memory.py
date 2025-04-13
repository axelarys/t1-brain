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

PROMOTE_THRESHOLD = 5
TIME_THRESHOLD = 86400  # 1 day


def sha256_hash(text):
    """
    Generate a SHA-256 hash from text.
    This function is used for deduplication and lookup.
    
    Args:
        text: The text to hash
        
    Returns:
        str: The SHA-256 hash as a hexadecimal string
    """
    return hashlib.sha256(text.encode()).hexdigest()

def chunk_by_tokens(text, max_tokens=300):
    """
    Split text into chunks based on token count.
    
    Args:
        text: The text to chunk
        max_tokens: Maximum tokens per chunk
        
    Returns:
        list: List of text chunks
    """
    words, chunks, current = text.split(), [], []
    for word in words:
        current.append(word)
        if len(encoder.encode(" ".join(current))) >= max_tokens:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks


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
        """
        Connect to PostgreSQL database.
        Ensures connection is active before database operations.
        """
        try:
            if not self.pg_conn or self.pg_conn.closed:
                self.pg_conn = psycopg2.connect(
                    host=settings.PG_HOST,
                    database=settings.PG_DATABASE,
                    user=settings.PG_USER,
                    password=settings.PG_PASSWORD
                )
                self.pg_cursor = self.pg_conn.cursor()
                session_logger.debug("✅ Connected to PostgreSQL database")
        except Exception as e:
            session_logger.error(f"❌ Database connection error: {e}")
            raise

    def generate_embedding(self, content):
        """
        Generate embedding vector for content using OpenAI API.
        
        Args:
            content: The text content to embed
            
        Returns:
            list: The embedding vector as a list of floats
        """
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
        """
        Determine if a memory should be promoted to PostgreSQL/FAISS.
        
        Args:
            memory: The memory object
            
        Returns:
            bool: True if the memory should be promoted, False otherwise
        """
        access_score = memory.get("access_score", 0)
        created_at = memory.get("timestamp", time.time())
        should_promote = access_score >= PROMOTE_THRESHOLD or (time.time() - created_at > TIME_THRESHOLD)
        
        if should_promote:
            session_logger.debug(f"🔍 Memory qualifies for promotion: score={access_score}, threshold={PROMOTE_THRESHOLD}")
        
        return should_promote

    def update_postgres(self, session_id, memory, sha, embedding=None):
        """
        Update or insert memory data in PostgreSQL.
        
        Args:
            session_id: The session identifier
            memory: The memory object
            sha: SHA hash of the memory content
            embedding: Optional embedding vector for new entries
            
        Returns:
            bool: True if operation succeeded, False otherwise
        """
        try:
            self.connect_to_db()
            now_ts = datetime.now().isoformat()
            embedding_str = None

            # Convert embedding to string format if provided
            if embedding:
                if isinstance(embedding, list):
                    embedding_str = "[" + ",".join(map(str, embedding)) + "]"
                elif isinstance(embedding, str):
                    embedding_str = embedding  # Already in string form
                else:
                    raise ValueError(f"Embedding format invalid: {type(embedding)}")

            access_score = memory.get("access_score", 0)
            last_accessed = memory.get("last_accessed", time.time())

            # First check if the record exists
            self.pg_cursor.execute("SELECT 1 FROM embeddings WHERE sha_hash = %s", (sha,))
            record_exists = self.pg_cursor.fetchone() is not None

            if record_exists:
                # Update existing record
                sql = """
                    UPDATE embeddings 
                    SET access_score = %s, last_accessed = to_timestamp(%s)
                    WHERE sha_hash = %s
                    RETURNING id, access_score
                """
                params = (access_score, last_accessed, sha)
            else:
                # Insert new record
                sql = """
                    INSERT INTO embeddings (
                        session_id, query, response, embedding, memory_type, sentiment,
                        source_type, image_url, sha_hash, access_score, last_accessed, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, to_timestamp(%s), NOW())
                    RETURNING id, access_score
                """
                params = (
                    session_id,
                    memory["query"],
                    memory["response"],
                    embedding_str,
                    memory.get("memory_type", "semantic"),
                    memory.get("sentiment", "neutral"),
                    memory.get("source_type", "text"),
                    memory.get("image_url", None),
                    sha,
                    access_score,
                    last_accessed
                )
            
            # Execute SQL with parameters
            self.pg_cursor.execute(sql, params)
            
            # Get the result of the operation
            result = self.pg_cursor.fetchone()
            
            # Commit the transaction
            self.pg_conn.commit()
            
            if result:
                pg_id, pg_score = result
                session_logger.info(f"✅ PostgreSQL updated: id={pg_id}, sha={sha[:10]}, score={pg_score}")
                
                # Add to FAISS if embedding was provided
                if embedding and hasattr(self, '_warm_cache'):
                    self._warm_cache.add(
                        session_id,
                        memory["query"],
                        memory["response"],
                        embedding,
                        access_score=access_score,
                        last_accessed=last_accessed
                    )
                    session_logger.info(f"🔥 FAISS updated for sha={sha[:10]}")
                
                return True
            else:
                session_logger.warning(f"⚠️ PostgreSQL update returned no result for sha={sha[:10]}")
                return False
                
        except Exception as e:
            # Roll back transaction on error
            if self.pg_conn:
                self.pg_conn.rollback()
                
            # Log detailed error information
            session_logger.error(f"❌ PostgreSQL update error for sha={sha[:10]}: {e}")
            
            # Log PostgreSQL-specific error details if available
            if isinstance(e, psycopg2.Error):
                if hasattr(e, 'pgerror') and e.pgerror:
                    session_logger.error(f"PostgreSQL error details: {e.pgerror}")
                if hasattr(e, 'diag') and e.diag:
                    session_logger.error(f"PostgreSQL diagnostic: {e.diag.message_primary}")
            
            return False

    def store_memory(self, session_id, query, response, memory_type="semantic", sentiment="neutral", source_type="text", metadata=None):
        """
        Store memory in Redis and optionally promote to PostgreSQL/FAISS.
        
        Args:
            session_id: The session identifier
            query: The query text
            response: The response text
            memory_type: Type of memory (semantic, image, etc.)
            sentiment: Sentiment of the memory
            source_type: Source type of the memory
            metadata: Additional metadata
            
        Returns:
            dict: Status of the operation
        """
        try:
            self.connect_to_db()
            key = f"memory:{session_id}"
            self.redis_client.setex(f"session:{session_id}", self.ttl, json.dumps({"session_id": session_id}))

            is_image = memory_type == "image"
            image_url = None

            # Handle image processing if needed
            if is_image:
                try:
                    api_key = get_api_key("image")
                    client = OpenAI(api_key=api_key)
                    extraction = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe this image."},
                                {"type": "image_url", "image_url": {"url": query}}
                            ]
                        }]
                    )
                    query = extraction.choices[0].message.content.strip()
                    image_url = query
                except Exception as e:
                    session_logger.error(f"Image extraction failed: {e}")
                    return {"status": "error", "message": "Image processing failed"}

            # Chunk text if needed
            chunks = [query] if is_image else chunk_by_tokens(query) if len(encoder.encode(query)) > 400 else [query]
            session_logger.info(f"Storing memory in {len(chunks)} chunk(s) for session {session_id}")

            for i, chunk in enumerate(chunks):
                chunk_id = f"{session_id}_chunk_{i+1}"
                sha = sha256_hash(chunk + response)
                embedding = self.generate_embedding(chunk)
                now = time.time()

                # Create memory object
                memory = {
                    "query": chunk,
                    "response": response,
                    "timestamp": now,
                    "access_score": 0,
                    "last_accessed": now,
                    "chunk_id": chunk_id,
                    "memory_type": memory_type,
                    "sentiment": sentiment,
                    "source_type": source_type,
                    "image_url": image_url,
                    "memory_policy": {"short_term": memory_type != "summary", "expiration": 3 * 86400}
                }
                if metadata:
                    memory.update(metadata)

                # Store in Redis
                self.redis_client.rpush(key, json.dumps(memory))
                self.redis_client.expire(key, self.ttl)

                # Store in PostgreSQL/FAISS
                self.update_postgres(session_id, memory, sha, embedding)
                
                # Store in Glacier
                upload_object(f"{session_id}_{int(now)}.json", memory)

            return {"status": "stored", "message": f"Memory stored in {len(chunks)} chunk(s)."}

        except Exception as e:
            session_logger.error(f"Memory store failed: {e}")
            return {"status": "error", "message": "Memory store failed"}

    def retrieve_memory(self, session_id, query):
        """
        Retrieve memory from Redis, PostgreSQL/FAISS, or Glacier.
        
        Args:
            session_id: The session identifier
            query: The query text
            
        Returns:
            list: List of memory items
        """
        try:
            self.connect_to_db()
            key = f"memory:{session_id}"

            # Check Redis first (hot cache)
            if self.redis_client.exists(key):
                session_logger.info(f"🔍 Found memory in Redis for session {session_id}")
                memory_items = [json.loads(m) for m in self.redis_client.lrange(key, 0, -1)]
                updated = []
                promoted_count = 0

                # Process each memory item
                for m in memory_items:
                    # Update access score and timestamp
                    m["access_score"] = m.get("access_score", 0) + 1
                    m["last_accessed"] = time.time()
                    sha = sha256_hash(m["query"] + m["response"])

                    session_logger.debug(f"➡️ Memory: {m['query'][:40]} | Score={m['access_score']}")

                    # Check if memory should be promoted
                    if self.should_promote(m):
                        session_logger.info(f"🚀 Attempting promotion for: {m['query'][:40]}...")
                        embedding = self.generate_embedding(m["query"])
                        promotion_success = self.update_postgres(session_id, m, sha, embedding)
                        
                        if promotion_success:
                            promoted_count += 1
                            session_logger.info(f"🚀 Promoted to PostgreSQL/FAISS: {m['query'][:40]}...")
                        else:
                            session_logger.warning(f"⚠️ Promotion failed for: {m['query'][:40]}...")
                    else:
                        # Just update access stats, no promotion needed
                        self.update_postgres(session_id, m, sha)

                    updated.append(json.dumps(m))  # pre-serialize

                # Use Redis pipeline for atomic operations
                try:
                    pipe = self.redis_client.pipeline()
                    pipe.delete(key)
                    if updated:
                        pipe.rpush(key, *updated)
                        pipe.expire(key, self.ttl)
                    pipe.execute()
                    session_logger.info(f"🧠 Redis memory updated for {session_id} with {len(updated)} items")
                except redis.RedisError as e:
                    session_logger.error(f"❌ Redis pipeline failed: {e}")
                    # Fallback to individual operations
                    for m_json in updated:
                        try:
                            self.redis_client.rpush(key, m_json)
                        except Exception as redis_err:
                            session_logger.error(f"❌ Redis rpush failed: {redis_err}")
                    self.redis_client.expire(key, self.ttl)

                if promoted_count > 0:
                    session_logger.info(f"✅ Successfully promoted {promoted_count} memories to PostgreSQL/FAISS")

                return [json.loads(m) for m in updated]

            # FAISS fallback (warm cache)
            session_logger.info(f"🔍 No memory in Redis, checking FAISS for session {session_id}")
            embedding = self.generate_embedding(query)
            warm_hits = self._warm_cache.find_similar(embedding)
            if warm_hits:
                session_logger.info(f"🔥 FAISS fallback returned {len(warm_hits)} result(s)")
                for hit in warm_hits:
                    hit["access_score"] = hit.get("access_score", 0) + 1
                    hit["last_accessed"] = time.time()
                    sha = sha256_hash(hit["query"] + hit["response"])
                    self.update_postgres(session_id, hit, sha)
                return warm_hits

            # Glacier fallback (cold storage)
            session_logger.info(f"🔍 No memory in FAISS, checking Glacier for session {session_id}")
            glacier_key = f"{session_id}_glacier.json"
            glacier_blob = download_object(glacier_key)
            if glacier_blob:
                try:
                    glacier_data = json.loads(glacier_blob) if isinstance(glacier_blob, str) else glacier_blob
                    if isinstance(glacier_data, dict):
                        glacier_data = [glacier_data]

                    # Promote and rehydrate
                    for m in glacier_data:
                        m["access_score"] = m.get("access_score", 0) + 1
                        m["last_accessed"] = time.time()
                        self.redis_client.rpush(key, json.dumps(m))
                    self.redis_client.expire(key, self.ttl)

                    session_logger.info(f"❄️ Rehydrated {len(glacier_data)} items from Glacier into Redis")
                    return glacier_data

                except Exception as e:
                    session_logger.error(f"❌ Glacier fallback error: {e}")
                    return []

            session_logger.warning("⚠️ No memory found in Redis, FAISS, or Glacier.")
            return []

        except Exception as e:
            session_logger.error(f"❌ Retrieval error: {e}")
            import traceback
            session_logger.error(traceback.format_exc())
            return []

    def run_consistency_check(self):
        """
        Check consistency between PostgreSQL and FAISS.
        Ensures that all entries in PostgreSQL are also in FAISS.
        
        Returns:
            dict: Status of the operation
        """
        try:
            self.connect_to_db()
            self.pg_cursor.execute("SELECT session_id, query, response, embedding, sha_hash, access_score FROM embeddings")
            records = self.pg_cursor.fetchall()
            fixed = 0
            errors = 0

            session_logger.info(f"Running consistency check on {len(records)} PostgreSQL records")

            for record in records:
                sid, q, r, emb_str, sha, score = record
                try:
                    # Parse embedding from string
                    if emb_str and emb_str.startswith('[') and emb_str.endswith(']'):
                        emb = json.loads(emb_str)
                        
                        # Check if entry exists in FAISS
                        if not self._warm_cache.exists(sid, q, r):
                            # Add to FAISS if not exists
                            self._warm_cache.add(sid, q, r, emb, access_score=score, last_accessed=time.time())
                            fixed += 1
                            session_logger.info(f"✅ Added missing entry to FAISS: {q[:40]}...")
                except Exception as e:
                    session_logger.warning(f"Skip FAISS sync error for {sha}: {e}")
                    errors += 1

            # Save FAISS index if changes were made
            if fixed > 0:
                self._warm_cache.save_index()

            return {
                "status": "done", 
                "faiss_added": fixed, 
                "total_records": len(records),
                "errors": errors
            }

        except Exception as e:
            session_logger.error(f"❌ Consistency check failed: {e}")
            import traceback
            session_logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}
            
    def debug_promotion_flow(self, session_id, memory_index=0):
        """
        Debug helper to trace the promotion flow for a specific memory item.
        
        Args:
            session_id: The session identifier
            memory_index: Index of the memory item to debug (default: 0 for first item)
            
        Returns:
            dict: Debug information about the promotion process
        """
        try:
            key = f"memory:{session_id}"
            if not self.redis_client.exists(key):
                return {"status": "error", "message": f"No memory found for session {session_id}"}
                
            memory_items = [json.loads(m) for m in self.redis_client.lrange(key, 0, -1)]
            if not memory_items or memory_index >= len(memory_items):
                return {"status": "error", "message": f"Memory index {memory_index} out of range"}
                
            m = memory_items[memory_index]
            sha = sha256_hash(m["query"] + m["response"])
            
            # Check if should promote
            should_promote = self.should_promote(m)
            
            # Check if already in PostgreSQL
            self.connect_to_db()
            self.pg_cursor.execute("SELECT access_score, created_at FROM embeddings WHERE sha_hash = %s", (sha,))
            pg_record = self.pg_cursor.fetchone()
            
            # Check if in FAISS
            in_faiss = self._warm_cache.exists(session_id, m["query"], m["response"])
            
            # Generate debug info
            debug_info = {
                "status": "success",
                "memory": {
                    "query": m["query"],
                    "access_score": m["access_score"],
                    "last_accessed": m["last_accessed"],
                    "timestamp": m.get("timestamp", "unknown"),
                    "sha": sha
                },
                "promotion": {
                    "should_promote": should_promote,
                    "promote_threshold": PROMOTE_THRESHOLD,
                    "time_threshold": TIME_THRESHOLD,
                    "in_postgresql": pg_record is not None,
                    "in_faiss": in_faiss,
                    "pg_access_score": pg_record[0] if pg_record else None,
                    "pg_created_at": pg_record[1].isoformat() if pg_record and pg_record[1] else None
                }
            }
            
            return debug_info
            
        except Exception as e:
            session_logger.error(f"❌ Debug promotion error: {e}")
            import traceback
            session_logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}
