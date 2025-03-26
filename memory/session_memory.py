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

# Ensure Python can locate settings.py
sys.path.append("/root/t1-brain/config")

# Import configurations from settings.py
from settings import OPENAI_API_KEY, PG_HOST, PG_DATABASE, PG_USER, PG_PASSWORD

# Configure Logging
logging.basicConfig(
    filename="/root/t1-brain/logs/session_memory.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY


class PersistentSessionMemory:
    def __init__(self, redis_host="localhost", redis_port=6379, redis_db=0, ttl=1800):
        """Initialize Redis and PostgreSQL connections."""
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
        self.ttl = ttl

    def connect_to_db(self):
        """Ensure PostgreSQL connection is active."""
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
        """Generate embedding using OpenAI API with fallback."""
        try:
            response = openai.embeddings.create(
                model="text-embedding-ada-002", input=[query]
            )
            embedding = response.data[0].embedding
            if len(embedding) != 1536:
                raise ValueError("Embedding dimension mismatch. Expected 1536.")
            return embedding
        except Exception as e:
            logging.error(f"❌ Error generating embedding: {e}")
            return [0.0] * 1536  # Default embedding with correct dimensions

    def store_memory(self, session_id, query, response, memory_type="semantic", sentiment="neutral"):
        """Store memory with Redis caching and PostgreSQL backup."""
        self.connect_to_db()

        session_key = f"session:{session_id}"
        if not self.redis_client.exists(session_key):
            self.redis_client.setex(session_key, self.ttl, json.dumps({"session_id": session_id}))
            logging.info(f"✅ Persistent session created: {session_id}")

        data = json.dumps({
            "query": query, "response": response, "timestamp": time.time(),
            "memory_type": memory_type, "sentiment": sentiment
        })
        self.redis_client.rpush(f"memory:{session_id}", data)
        self.redis_client.expire(f"memory:{session_id}", self.ttl)

        embedding = self.generate_embedding(query)
        embedding_str = "[" + ",".join(map(str, embedding)) + "]"

        try:
            self.pg_cursor.execute(
                """
                INSERT INTO embeddings (session_id, query, response, embedding, memory_type, sentiment, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
                """,
                (session_id, query, response, embedding_str, memory_type, sentiment)
            )
            self.pg_conn.commit()
            return {"status": "stored", "message": "Memory successfully stored."}
        except Exception as e:
            logging.error(f"❌ PostgreSQL Insertion Error: {e}")
            return {"status": "error", "message": "Database error occurred."}

    def delete_memory(self, session_id, query):
        """Delete memory safely from PostgreSQL and Redis."""
        self.connect_to_db()
        try:
            self.pg_cursor.execute(
                "DELETE FROM embeddings WHERE session_id = %s AND query = %s",
                (session_id, query)
            )
            self.pg_conn.commit()
            self.redis_client.delete(f"memory:{session_id}")
            return {"status": "deleted", "message": "Memory successfully deleted."}
        except Exception as e:
            logging.error(f"❌ Memory Deletion Error: {e}")
            return {"status": "error", "message": "Database error occurred."}
