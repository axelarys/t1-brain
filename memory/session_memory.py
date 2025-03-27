import sys
import os
import redis
import psycopg2
import json
import time
import openai
import logging
from datetime import datetime
from fastapi import HTTPException

# Add config + memory paths
sys.path.append("/root/projects/t1-brain/config")
sys.path.append("/root/projects/t1-brain/memory")

from settings import OPENAI_API_KEY, PG_HOST, PG_DATABASE, PG_USER, PG_PASSWORD
from graph_memory import GraphMemory

# Logging setup
LOG_DIR = "/root/projects/t1-brain/logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "session_memory.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Set OpenAI key
openai.api_key = OPENAI_API_KEY

class PersistentSessionMemory:
    def __init__(self, redis_host="localhost", redis_port=6379, redis_db=0, ttl=1800):
        self.ttl = ttl
        self.graph_memory = GraphMemory()

        try:
            self.redis_client = redis.StrictRedis(
                host=redis_host, port=redis_port, db=redis_db, decode_responses=True
            )
            self.redis_client.ping()
            logging.info("✅ Redis connection established.")
        except redis.ConnectionError:
            logging.error("❌ Redis connection failed.")
            self.redis_client = None

        self.connect_to_db()

    def connect_to_db(self):
        try:
            self.pg_conn = psycopg2.connect(
                host=PG_HOST, database=PG_DATABASE, user=PG_USER, password=PG_PASSWORD
            )
            self.pg_cursor = self.pg_conn.cursor()
            logging.info("✅ PostgreSQL connection established.")
        except psycopg2.OperationalError as e:
            logging.error(f"❌ PostgreSQL Connection Error: {e}")
            self.pg_conn = None

    def generate_embedding(self, query):
        try:
            response = openai.embeddings.create(
                model="text-embedding-ada-002", input=[query]
            )
            embedding = response.data[0].embedding
            if len(embedding) != 1536:
                raise ValueError("Embedding dimension mismatch.")
            return embedding
        except Exception as e:
            logging.error(f"❌ Embedding error: {e}")
            return [0.0] * 1536

    def store_memory(self, session_id, query, response, memory_type="semantic", sentiment="neutral"):
        self.connect_to_db()

        # Redis store
        try:
            session_key = f"session:{session_id}"
            if not self.redis_client.exists(session_key):
                self.redis_client.setex(session_key, self.ttl, json.dumps({"session_id": session_id}))
                logging.info(f"✅ New session key: {session_id}")

            memory_data = json.dumps({
                "query": query, "response": response, "timestamp": time.time(),
                "memory_type": memory_type, "sentiment": sentiment
            })
            self.redis_client.rpush(f"memory:{session_id}", memory_data)
            self.redis_client.expire(f"memory:{session_id}", self.ttl)
        except Exception as e:
            logging.error(f"❌ Redis error: {e}")

        # Vector DB (Postgres)
        try:
            embedding = self.generate_embedding(query)
            embedding_str = "[" + ",".join(map(str, embedding)) + "]"
            self.pg_cursor.execute(
                """
                INSERT INTO embeddings (session_id, query, response, embedding, memory_type, sentiment, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
                """,
                (session_id, query, response, embedding_str, memory_type, sentiment)
            )
            self.pg_conn.commit()
        except Exception as e:
            logging.error(f"❌ PostgreSQL error: {e}")
            return {"status": "error", "message": "PostgreSQL insert failed"}

        # Graph DB (Neo4j)
        try:
            self.graph_memory.store_graph_memory(session_id, query, response, memory_type, sentiment)
        except Exception as e:
            logging.error(f"❌ GraphMemory error: {e}")

        return {"status": "stored", "message": "Memory stored in Redis, Postgres and Graph"}

    def retrieve_memory(self, session_id, query=None):
        try:
            key = f"memory:{session_id}"
            if not self.redis_client.exists(key):
                logging.info(f"ℹ️ No session found: {session_id}")
                return []

            entries = self.redis_client.lrange(key, 0, -1)
            parsed = [json.loads(e) for e in entries]

            return [p for p in parsed if p["query"] == query] if query else parsed
        except Exception as e:
            logging.error(f"❌ Retrieval error: {e}")
            return []

    def delete_memory(self, session_id, query):
        self.connect_to_db()
        try:
            self.pg_cursor.execute(
                "DELETE FROM embeddings WHERE session_id = %s AND query = %s",
                (session_id, query)
            )
            self.pg_conn.commit()
            self.redis_client.delete(f"memory:{session_id}")
            return {"status": "deleted", "message": "Deleted from Redis and Postgres"}
        except Exception as e:
            logging.error(f"❌ Delete error: {e}")
            return {"status": "error", "message": "Delete failed"}
