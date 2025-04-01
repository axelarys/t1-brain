import httpx
import json
import time
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# URL of your running FastAPI app
BASE_URL = "http://localhost:8000"
HEALTH_URL = f"{BASE_URL}/health"
API_URL = f"{BASE_URL}/memory/store_image"

# Test payload
payload = {
    "session_id": "test_001",
    "image_url": "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png",
    "response": "User uploaded a reference image for demo",
    "sentiment": "neutral"
}

headers = {
    "Content-Type": "application/json",
    "X-API-KEY": "WsRocks1234"  # make sure this matches the expected key
}

# First test the health endpoint
logger.info("Testing health endpoint...")
try:
    response = httpx.get(HEALTH_URL, timeout=5)
    logger.info(f"Health check: {response.status_code}")
    logger.info(f"Response: {response.json()}")
except Exception as e:
    logger.error(f"Health check failed: {str(e)}")

time.sleep(1)

# Then test the actual endpoint
logger.info(f"Testing image storage endpoint: {API_URL}")
logger.info(f"Headers: {headers}")
logger.info(f"Payload: {json.dumps(payload, indent=2)}")

try:
    # Use httpx with debugging and longer timeout
    client = httpx.Client(timeout=30.0)
    response = client.post(API_URL, json=payload, headers=headers)
    
    logger.info(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        logger.info(f"Response: {response.json()}")
    else:
        logger.info(f"Response Text: {response.text}")
except Exception as e:
    logger.error(f"Request failed: {str(e)}")
    logger.error(f"Error type: {type(e).__name__}")
    logger.error(f"Error details: {str(e)}")
