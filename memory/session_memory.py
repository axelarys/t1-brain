import os, json, time, redis, logging, psycopg2, hashlib, numpy as np, tiktoken
from config import settings
from openai import OpenAI
from utils.memory_utils import get_api_key

# 📂 Logging setup
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

# 🔢 Tokenizer
encoder = tiktoken.encoding_for_model("text-embedding-ada-002")

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
        self._graph_memory = None  # Lazy init

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
            valid_memory = []
            removed_count = 0

            all_memory = self.redis_client.lrange(key, 0, -1)
            self.redis_client.delete(key)  ## clear list to re-insert only valid ones

            for m in all_memory:
                try:
                    m_obj = json.loads(m)
                    policy = m_obj.get("memory_policy", {})
                    expiration = policy.get("expiration", 0)
                    short_term = policy.get("short_term", False)
                    timestamp = m_obj.get("timestamp", 0)

                    ## if expired short-term → skip
                    if short_term and expiration and now > (timestamp + expiration):
                        removed_count += 1
                        continue

                    valid_memory.append(json.dumps(m_obj))
                except Exception as e:
                    session_logger.warning(f"⚠️ Failed parsing memory item: {e}")

            ## re-push valid ones
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

    def store_memory(self, session_id, query, response, memory_type="semantic", sentiment="neutral"):
        self.connect_to_db()
        try:
            session_key = f"session:{session_id}"
            if not self.redis_client.exists(session_key):
                self.redis_client.setex(session_key, self.ttl, json.dumps({"session_id": session_id}))
                session_logger.info(f"✅ New session key: {session_id}")

            ## Cleanup short-term expired memory before storing
            self.cleanup_expired_memory(session_id)

            is_image = memory_type == "image"
            if is_image:
                try:
                    api_key = get_api_key("image")
                    client = OpenAI(api_key=api_key)
                    extraction = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "user", "content": [
                                {"type": "text", "text": "Describe this image for memory embedding."},
                                {"type": "image_url", "image_url": {"url": query}}
                            ]}
                        ]
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
                embedding_str = "[" + ",".join(map(str, embedding)) + "]"

                memory_metadata = {
                    "query": chunk,
                    "response": response,
                    "timestamp": time.time(),
                    "memory_type": memory_type,
                    "sentiment": sentiment,
                    "chunk_id": chunk_id,
                    "source_type": "image" if is_image else "text",
                    "image_url": image_url if is_image else None,
                    "memory_policy": {
                        "short_term": memory_type != "summary",
                        "expiration": 3 * 86400 if memory_type != "summary" else 0
                    }
                }

                self.redis_client.rpush(f"memory:{session_id}", json.dumps(memory_metadata))
                self.redis_client.expire(f"memory:{session_id}", self.ttl)

                self.pg_cursor.execute(
                    """INSERT INTO embeddings (session_id, query, response, embedding, memory_type, sentiment, source_type, image_url, sha_hash, created_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())""",
                    (
                        session_id,
                        chunk,
                        response,
                        embedding_str,
                        memory_type,
                        sentiment,
                        "image" if is_image else "text",
                        image_url if is_image else None,
                        dedup_hash
                    )
                )
                self.pg_conn.commit()

                self.graph_memory.store_graph_memory(session_id, chunk, response, memory_type, sentiment)

        except Exception as e:
            session_logger.error(f"❌ Store error: {e}")
            return {"status": "error", "message": "Memory store failed"}

        return {"status": "stored", "message": f"Memory stored in {len(chunks)} chunk(s)."}

    def retrieve_memory(self, session_id, query):
        try:
            key = f"memory:{session_id}"
            if self.redis_client.exists(key):
                return [json.loads(m) for m in self.redis_client.lrange(key, 0, -1)]
            return []
        except Exception as e:
            session_logger.error(f"❌ Retrieval error: {e}")
            return []

    def delete_memory(self, session_id, query):
        try:
            self.redis_client.delete(f"memory:{session_id}")
            self.pg_cursor.execute("DELETE FROM embeddings WHERE session_id = %s", (session_id,))
            self.pg_conn.commit()
            return {"status": "deleted", "message": "Session memory deleted."}
        except Exception as e:
            session_logger.error(f"❌ Deletion error: {e}")
            return {"status": "error", "message": "Memory deletion failed"}

    def find_similar_queries(self, query, top_k=3):
        try:
            self.connect_to_db()
            embedding = self.generate_embedding(query)
            embedding_str = "[" + ",".join(map(str, embedding)) + "]"
            self.pg_cursor.execute(
                """SELECT session_id, query, response, 1 - (embedding <=> %s) AS similarity
                   FROM embeddings ORDER BY embedding <=> %s LIMIT %s""",
                (embedding_str, embedding_str, top_k)
            )
            results = self.pg_cursor.fetchall()
            return [{"session_id": r[0], "query": r[1], "response": r[2], "similarity": float(r[3])} for r in results]
        except Exception as e:
            session_logger.error(f"❌ Similarity search error: {e}")
            return []

    def restore_session_from_pg(self, session_id):
        self.connect_to_db()
        try:
            self.pg_cursor.execute("SELECT query, response, memory_type, sentiment, source_type, image_url FROM embeddings WHERE session_id = %s", (session_id,))
            rows = self.pg_cursor.fetchall()

            if not rows:
                return {"status": "error", "message": "No memory found for session."}

            key = f"memory:{session_id}"
            self.redis_client.delete(key)

            ## Cleanup first in case of remnants
            self.cleanup_expired_memory(session_id)

            for i, row in enumerate(rows):
                query, response, memory_type, sentiment, source_type, image_url = row
                memory = {
                    "query": query,
                    "response": response,
                    "timestamp": time.time(),
                    "memory_type": memory_type,
                    "sentiment": sentiment,
                    "chunk_id": f"{session_id}_chunk_{i+1}",
                    "source_type": source_type,
                    "image_url": image_url,
                    "memory_policy": {
                        "short_term": memory_type != "summary",
                        "expiration": 3 * 86400 if memory_type != "summary" else 0
                    }
                }
                self.redis_client.rpush(key, json.dumps(memory))

            self.redis_client.expire(key, self.ttl)
            session_logger.info(f"♻️ Restored session '{session_id}' into Redis with {len(rows)} items.")
            return {"status": "restored", "count": len(rows)}

        except Exception as e:
            session_logger.error(f"❌ Session restore error: {e}")
            return {"status": "error", "message": "Restore failed"}

    def summarize_session(self, session_id):
        self.connect_to_db()
        try:
            memories = self.retrieve_memory(session_id, "")
            if not memories:
                return {"status": "error", "message": "No memory found to summarize."}

            summary_input = "\n".join([f"User: {m['query']}\nAssistant: {m['response']}" for m in memories])
            token_logger.info(f"summarize_session | input_length={len(summary_input)}")

            api_key = get_api_key("text")
            client = OpenAI(api_key=api_key)

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Summarize the following conversation for memory recall:"},
                    {"role": "user", "content": summary_input}
                ]
            )

            summary_text = response.choices[0].message.content.strip()
            session_logger.info(f"📄 Session summary generated:\n{summary_text}")

            return self.store_memory(
                session_id=session_id,
                query="Session Summary",
                response=summary_text,
                memory_type="summary",
                sentiment="neutral"
            )

        except Exception as e:
            session_logger.error(f"❌ Summarization error: {e}")
            return {"status": "error", "message": "Summarization failed"}
