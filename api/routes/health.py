# api/routes/health.py

import os
import time
import logging
import redis
import psycopg2
from fastapi import APIRouter
from neo4j import GraphDatabase

logger = logging.getLogger("health")

health_router = APIRouter()
START_TIME = time.time()

@health_router.get("/healthcheck")
def healthcheck():
    status = {
        "status": "ok",
        "redis": "disconnected",
        "postgres": "disconnected",
        "neo4j": "disconnected",
        "uptime_sec": int(time.time() - START_TIME),
    }

    # Redis check
    try:
        r = redis.Redis(
            host="localhost",
            port=6379,
            decode_responses=True
        )
        if r.ping():
            status["redis"] = "connected"
    except Exception as e:
        logger.warning(f"[Health] Redis error: {e}")

    # PostgreSQL check
    try:
        conn = psycopg2.connect("postgresql://postgres:postgres1806@localhost:5432/t1brain_db")
        cur = conn.cursor()
        cur.execute("SELECT 1;")
        _ = cur.fetchone()
        status["postgres"] = "connected"
        cur.close()
        conn.close()
    except Exception as e:
        logger.warning(f"[Health] PostgreSQL error: {e}")

    # Neo4j check
    try:
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "neo4j1806"))
        with driver.session() as session:
            session.run("RETURN 1").consume()
        status["neo4j"] = "connected"
    except Exception as e:
        logger.warning(f"[Health] Neo4j error: {e}")

    return status
