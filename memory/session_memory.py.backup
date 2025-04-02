import os, json, time, redis, logging, psycopg2, numpy as np, tiktoken
from config import settings
from openai import OpenAI
from memory.graph_memory import GraphMemory
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

class PersistentSessionMemory:
    def __init__(self):
        self.ttl = 86400
        self.pg_conn = None
        self.pg_cursor = None
        self.redis_client = redis.StrictRedis(host="localhost", port=6379, decode_responses=True)
        self.graph_memory = GraphMemory()

    def connect_to_db(self):
        if not self.pg_conn or self.pg_conn.closed:
            self.pg_conn = psycopg2.connect(
                host=settings.PG_HOST,
                database=settings.PG_DATABASE,
                user=settings.PG_USER,
                password=settings.PG_PASSWORD
            )
            self.pg_cursor = self.pg_conn.cursor()

    def generate_embedding(self, content, memory_type="text"):
        try:
            api_key = get_api_key(memory_type)
            client = OpenAI(api_key=api_key)

            if memory_type == "image":
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "user", "content": [
                            {"type": "text", "text": "Understand and embed this image:"},
                            {"type": "image_url", "image_url": {"url": content}}
                        ]}
                    ]
                )
                return np.random.rand(1536).tolist()  # Placeholder
            else:
                response = client.embeddings.create(model="text-embedding-ada-002", input=content)
                usage = getattr(response, 'usage', None)
                if usage:
                    token_logger.info(f"embedding | tokens={usage.total_tokens} | model=text-embedding-ada-002")
                return response.data[0].embedding

        except Exception as e:
            session_logger.error(f"❌ Embedding error ({memory_type}): {e}")
            return np.zeros(1536).tolist()

    def store_memory(self, session_id, query, response, memory_type="semantic", sentiment="neutral"):
        self.connect_to_db()
        try:
            session_key = f"session:{session_id}"
            if not self.redis_client.exists(session_key):
                self.redis_client.setex(session_key, self.ttl, json.dumps({"session_id": session_id}))
                session_logger.info(f"✅ New session key: {session_id}")

            is_image = memory_type == "image"

            if is_image:
                try:
                    api_key = get_api_key("image")
                    client = OpenAI(api_key=api_key)
                    extraction = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "user", "content": [
                                {"type": "text", "text": "Describe this image for knowledge embedding."},
                                {"type": "image_url", "image_url": {"url": query}}
                            ]}
                        ]
                    )
                    query = extraction.choices[0].message.content.strip()
                    session_logger.info(f"🧠 Image description extracted: {query}")
                except Exception as e:
                    session_logger.error(f"❌ Image extraction failed: {e}")
                    return {"status": "error", "message": "Image analysis failed"}

            input_chunks = [query] if is_image else chunk_by_tokens(query) if len(encoder.encode(query)) > 400 else [query]

            for i, chunk in enumerate(input_chunks):
                self.pg_cursor.execute(
                    """SELECT 1 FROM embeddings WHERE session_id = %s AND query = %s AND response = %s LIMIT 1""",
                    (session_id, chunk, response)
                )
                if self.pg_cursor.fetchone():
                    session_logger.info(f"⚠️ Duplicate skipped (chunk: {chunk[:50]})")
                    continue

                chunk_id = f"{session_id}_chunk_{i+1}"
                memory_data = json.dumps({
                    "query": chunk,
                    "response": response,
                    "timestamp": time.time(),
                    "memory_type": memory_type,
                    "sentiment": sentiment,
                    "chunk_id": chunk_id
                })

                self.redis_client.rpush(f"memory:{session_id}", memory_data)
                self.redis_client.expire(f"memory:{session_id}", self.ttl)

                embedding = self.generate_embedding(chunk, memory_type=memory_type)
                embedding_str = "[" + ",".join(map(str, embedding)) + "]"

                self.pg_cursor.execute(
                    """INSERT INTO embeddings (session_id, query, response, embedding, memory_type, sentiment, created_at)
                       VALUES (%s, %s, %s, %s, %s, %s, NOW())""",
                    (session_id, chunk, response, embedding_str, memory_type, sentiment)
                )
                self.pg_conn.commit()

            self.graph_memory.store_graph_memory(session_id, query, response, memory_type, sentiment)

        except Exception as e:
            session_logger.error(f"❌ Store error: {e}")
            return {"status": "error", "message": "Memory store failed"}

        return {"status": "stored", "message": f"Memory stored in {len(input_chunks)} chunk(s)."}

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
