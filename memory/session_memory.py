# session_memory.py

import os, json, time, redis, logging, psycopg2, hashlib, numpy as np, tiktoken
from config import settings
from openai import OpenAI
from utils.memory_utils import get_api_key
from memory.warm_layer import WarmMemoryCache  # ✅ Warm layer support

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
        self._warm_cache = WarmMemoryCache.get_instance()  # ✅ Eager singleton load

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

    def store_memory(self, session_id, query, response, memory_type="semantic",
                     sentiment="neutral", source_type="text", metadata=None):
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
                    self._warm_cache.add(session_id, chunk, response, embedding)
                    session_logger.info(f"🔥 FAISS Add: session={session_id}, chunk={chunk[:50]}..., index_total={self._warm_cache.index.ntotal}")

                embedding_str = "[" + ",".join(map(str, embedding)) + "]"

                memory_metadata = {
                    "query": chunk,
                    "response": response,
                    "timestamp": time.time(),
                    "memory_type": memory_type,
                    "sentiment": sentiment,
                    "chunk_id": chunk_id,
                    "source_type": "image" if is_image else source_type,
                    "image_url": image_url if is_image else None,
                    "memory_policy": {
                        "short_term": memory_type != "summary",
                        "expiration": 3 * 86400 if memory_type != "summary" else 0
                    }
                }

                if metadata:
                    memory_metadata.update(metadata)

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
                        memory_metadata["source_type"],
                        memory_metadata["image_url"],
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

            embedding = self.generate_embedding(query)
            warm_hits = self._warm_cache.find_similar(embedding)

            if warm_hits:
                session_logger.info(f"🔥 FAISS fallback returned {len(warm_hits)} result(s)")
                for hit in warm_hits:
                    session_logger.info(f"➡️ Match | dist={hit.get('distance', 0):.4f} | query={hit['query']}")
                return warm_hits

            session_logger.warning("⚠️ FAISS fallback hit but returned no matches.")
            return []

        except Exception as e:
            session_logger.error(f"❌ Retrieval error: {e}")
            return []
