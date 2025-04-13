# memory/glacier_client.py

import boto3
import json
import logging
from botocore.exceptions import ClientError

# Setup logging
logger = logging.getLogger("glacier")
logging.basicConfig(level=logging.INFO)

# MinIO Config
GLACIER_BUCKET = "glacier-memory"
GLACIER_ENDPOINT = "http://localhost:19001"
GLACIER_ACCESS_KEY = "minioadmin"
GLACIER_SECRET_KEY = "minioadmin123"

# Boto3 S3 client
s3 = boto3.client(
    "s3",
    endpoint_url=GLACIER_ENDPOINT,
    aws_access_key_id=GLACIER_ACCESS_KEY,
    aws_secret_access_key=GLACIER_SECRET_KEY,
    region_name="us-east-1"
)

def upload_object(object_key: str, data: dict) -> bool:
    """Upload a JSON-serializable object to MinIO"""
    try:
        body = json.dumps(data).encode("utf-8")
        s3.put_object(Bucket=GLACIER_BUCKET, Key=object_key, Body=body)
        logger.info(f"[GLACIER] ✅ Uploaded: {object_key}")
        return True
    except ClientError as e:
        logger.error(f"[GLACIER] ❌ Upload error: {e}")
        return False

def download_object(object_key: str) -> dict:
    """Download a JSON object from MinIO"""
    try:
        obj = s3.get_object(Bucket=GLACIER_BUCKET, Key=object_key)
        content = obj["Body"].read().decode("utf-8")
        logger.info(f"[GLACIER] 📥 Downloaded: {object_key}")
        return json.loads(content)
    except ClientError as e:
        logger.warning(f"[GLACIER] ⚠️ Object not found: {object_key}")
        return {}
