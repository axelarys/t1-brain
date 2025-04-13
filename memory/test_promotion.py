#!/usr/bin/env python3
"""
Test script for verifying the memory promotion fix in T1 Brain.

This script tests the memory promotion flow by:
1. Creating a test memory entry in Redis
2. Retrieving it multiple times to increase access_score
3. Verifying promotion to PostgreSQL and FAISS
4. Providing detailed diagnostics about the process
"""

import os
import sys
import json
import time
import logging
import argparse
import psycopg2
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("promotion_test.log")
    ]
)
logger = logging.getLogger(__name__)

def verify_postgres_connection(config):
    """Verify PostgreSQL connection and table structure."""
    try:
        conn = psycopg2.connect(
            host=config["PG_HOST"],
            database=config["PG_DATABASE"],
            user=config["PG_USER"],
            password=config["PG_PASSWORD"]
        )
        cursor = conn.cursor()
        
        # Check if embeddings table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'embeddings'
            );
        """)
        table_exists = cursor.fetchone()[0]
        
        if not table_exists:
            logger.error("❌ 'embeddings' table does not exist in PostgreSQL!")
            return False
            
        # Check table structure
        cursor.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'embeddings';
        """)
        columns = cursor.fetchall()
        
        required_columns = {
            'id', 'session_id', 'query', 'response', 'embedding', 
            'sha_hash', 'access_score', 'last_accessed'
        }
        
        column_names = {col[0] for col in columns}
        missing_columns = required_columns - column_names
        
        if missing_columns:
            logger.error(f"❌ Missing required columns in 'embeddings' table: {missing_columns}")
            return False
            
        logger.info("✅ PostgreSQL connection and table structure verified")
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ PostgreSQL connection verification failed: {e}")
        return False

def verify_redis_connection():
    """Verify Redis connection."""
    try:
        import redis
        client = redis.StrictRedis(host="localhost", port=6379, decode_responses=True)
        ping_result = client.ping()
        
        if ping_result:
            logger.info("✅ Redis connection verified")
            return True
        else:
            logger.error("❌ Redis ping failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Redis connection verification failed: {e}")
        return False

def verify_faiss_index():
    """Verify FAISS index file exists and is accessible."""
    index_file = "/root/projects/t1-brain/warm_index.faiss"
    meta_file = "/root/projects/t1-brain/warm_metadata.pkl"
    
    index_exists = os.path.exists(index_file)
    meta_exists = os.path.exists(meta_file)
    
    if index_exists:
        logger.info(f"✅ FAISS index file exists: {index_file}")
    else:
        logger.warning(f"⚠️ FAISS index file does not exist: {index_file}")
        
    if meta_exists:
        logger.info(f"✅ FAISS metadata file exists: {meta_file}")
    else:
        logger.warning(f"⚠️ FAISS metadata file does not exist: {meta_file}")
        
    return index_exists or not meta_exists  # Return True if either file exists or both don't exist (new setup)

def check_postgres_for_session(config, session_id):
    """Check if session exists in PostgreSQL."""
    try:
        conn = psycopg2.connect(
            host=config["PG_HOST"],
            database=config["PG_DATABASE"],
            user=config["PG_USER"],
            password=config["PG_PASSWORD"]
        )
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT COUNT(*) FROM embeddings WHERE session_id = %s",
            (session_id,)
        )
        count = cursor.fetchone()[0]
        
        if count > 0:
            logger.info(f"✅ Found {count} records in PostgreSQL for session '{session_id}'")
        else:
            logger.warning(f"⚠️ No records found in PostgreSQL for session '{session_id}'")
            
        conn.close()
        return count
        
    except Exception as e:
        logger.error(f"❌ PostgreSQL check failed: {e}")
        return 0

def test_memory_promotion(session_id="test_promotion_fix", iterations=6, debug=False):
    """
    Test the memory promotion flow with the fixed implementation.
    
    Args:
        session_id: Unique identifier for the test session
        iterations: Number of retrieve operations to perform
        debug: Whether to run in debug mode with extra logging
    """
    try:
        # Import the session memory module
        sys.path.append("/root/projects/t1-brain")  # Adjust path as needed
        
        # Import configuration
        from config import settings
        
        # Verify connections before starting test
        pg_ok = verify_postgres_connection(settings.__dict__)
        redis_ok = verify_redis_connection()
        faiss_ok = verify_faiss_index()
        
        if not (pg_ok and redis_ok):
            logger.error("❌ Connection verification failed, aborting test")
            return False
            
        # Check if session already exists in PostgreSQL
        existing_records = check_postgres_for_session(settings.__dict__, session_id)
        if existing_records > 0:
            logger.warning(f"⚠️ Session '{session_id}' already exists in PostgreSQL with {existing_records} records")
            logger.warning(f"⚠️ Using a new session ID to avoid interference")
            session_id = f"{session_id}_{int(time.time())}"
            logger.info(f"✅ New session ID: {session_id}")
        
        # Set debug mode if requested
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.info("🔍 Debug mode enabled")
        
        # Import the session memory module
        from session_memory import PersistentSessionMemory
        
        # Create memory manager instance
        memory = PersistentSessionMemory()
        
        # Step 1: Store a test memory
        logger.info(f"Step 1: Storing test memory for session '{session_id}'")
        result = memory.store_memory(
            session_id=session_id,
            query="How does T1 Brain promotion testing work?",
            response="The promotion testing verifies that frequently accessed memories move from Redis to PostgreSQL and FAISS."
        )
        logger.info(f"Store result: {result}")
        
        if result.get("status") != "stored":
            logger.error(f"❌ Failed to store memory: {result}")
            return False
        
        # Step 2: Retrieve memory multiple times to increase access_score
        logger.info(f"Step 2: Retrieving memory {iterations} times to trigger promotion")
        for i in range(iterations):
            logger.info(f"Retrieval {i+1}/{iterations}")
            memories = memory.retrieve_memory(session_id, "How does T1 Brain promotion testing work?")
            if memories:
                logger.info(f"Retrieved memory with access_score: {memories[0]['access_score']}")
            else:
                logger.error("❌ No memories retrieved!")
                return False
            time.sleep(1)  # Small delay between retrievals
        
        # Step 3: Verify promotion to PostgreSQL
        logger.info("Step 3: Verifying promotion to PostgreSQL")
        memory.connect_to_db()
        memory.pg_cursor.execute(
            "SELECT id, query, access_score, created_at FROM embeddings WHERE session_id = %s",
            (session_id,)
        )
        pg_records = memory.pg_cursor.fetchall()
        
        if pg_records:
            logger.info(f"✅ Found {len(pg_records)} records in PostgreSQL:")
            for record in pg_records:
                logger.info(f"  - ID: {record[0]}, Query: {record[1][:30]}..., Access Score: {record[2]}, Created: {record[3]}")
            
            # Step 4: Run the debug helper for additional insights
            if hasattr(memory, 'debug_promotion_flow'):
                logger.info("Step 4: Running debug helper for additional insights")
                debug_info = memory.debug_promotion_flow(session_id)
                logger.info(f"Debug info: {json.dumps(debug_info, indent=2)}")
            
            # Step 5: Verify FAISS integration
            logger.info("Step 5: Verifying FAISS integration")
            if hasattr(memory._warm_cache, 'exists'):
                memories = memory.retrieve_memory(session_id, "How does T1 Brain promotion testing work?")
                if memories:
                    query = memories[0]["query"]
                    response = memories[0]["response"]
                    in_faiss = memory._warm_cache.exists(session_id, query, response)
                    if in_faiss:
                        logger.info("✅ Memory exists in FAISS")
                    else:
                        logger.warning("⚠️ Memory does not exist in FAISS")
            else:
                logger.warning("⚠️ FAISS exists() method not available")
            
            return True
        else:
            logger.error("❌ No records found in PostgreSQL after promotion attempts!")
            
            # Run debug helper if available
            if hasattr(memory, 'debug_promotion_flow'):
                debug_info = memory.debug_promotion_flow(session_id)
                logger.info(f"Debug info: {json.dumps(debug_info, indent=2)}")
            
            return False
    
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test T1 Brain memory promotion fix")
    parser.add_argument("--session-id", default=f"test_fix_{int(time.time())}", help="Session ID for testing")
    parser.add_argument("--iterations", type=int, default=6, help="Number of retrieval iterations")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    logger.info(f"Starting promotion test with session '{args.session_id}'")
    success = test_memory_promotion(args.session_id, args.iterations, args.debug)
    
    if success:
        logger.info("✅ Test PASSED: Memory promotion is working correctly!")
        sys.exit(0)
    else:
        logger.error("❌ Test FAILED: Memory promotion is not working correctly!")
        sys.exit(1)
