import sys
import time
import jwt
import redis
import logging
import psycopg2
from fastapi import Depends, HTTPException, Security
from fastapi.security import OAuth2PasswordBearer
from datetime import timedelta

# Ensure Python can locate settings.py
sys.path.append("/root/t1-brain/config")
from settings import PG_HOST, PG_DATABASE, PG_USER, PG_PASSWORD

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# JWT Configurations
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # ⏳ Access Token Expiry (1 Hour)
REFRESH_TOKEN_EXPIRE_DAYS = 7  # 🔄 Refresh Token Expiry (7 Days)

# Redis Connection
redis_client = redis.StrictRedis(host="localhost", port=6379, decode_responses=True)

# FastAPI Security
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

def verify_api_key(api_key: str):
    """Verify API Key from PostgreSQL before issuing JWT."""
    try:
        conn = psycopg2.connect(host=PG_HOST, database=PG_DATABASE, user=PG_USER, password=PG_PASSWORD)
        cursor = conn.cursor()
        cursor.execute("SELECT EXISTS (SELECT 1 FROM api_keys WHERE key = %s)", (api_key,))
        result = cursor.fetchone()[0]
        conn.close()
        return result
    except Exception as e:
        logging.error(f"❌ API Key Verification Failed: {str(e)}")
        return False

def create_jwt_token(data: dict, expires_delta: timedelta):
    """Generate a JWT token with expiration"""
    to_encode = data.copy()
    expire_timestamp = time.time() + expires_delta.total_seconds()
    to_encode.update({"exp": int(expire_timestamp)})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def generate_tokens(api_key: str):
    """Generate access & refresh tokens and store them in Redis"""
    redis_client.delete(f"session:{api_key}")
    redis_client.delete(f"refresh:{api_key}")

    access_token = create_jwt_token({"api_key": api_key}, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    refresh_token = create_jwt_token({"api_key": api_key}, timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS))

    redis_client.setex(f"session:{api_key}", ACCESS_TOKEN_EXPIRE_MINUTES * 60, access_token)
    redis_client.setex(f"refresh:{api_key}", REFRESH_TOKEN_EXPIRE_DAYS * 86400, refresh_token)

    return access_token, refresh_token

def verify_token(token: str = Depends(oauth2_scheme)):
    """Verify if a JWT token is valid"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def refresh_access_token(refresh_token: str):
    """Refresh an access token if the refresh token is still valid"""
    try:
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        api_key = payload.get("api_key")

        if redis_client.exists(f"refresh:{api_key}"):
            new_access_token = create_jwt_token({"api_key": api_key}, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
            redis_client.setex(f"session:{api_key}", ACCESS_TOKEN_EXPIRE_MINUTES * 60, new_access_token)
            return new_access_token
        else:
            raise HTTPException(status_code=401, detail="Session expired. Please reauthenticate.")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Refresh token expired. Please reauthenticate.")

def logout(api_key: str):
    """Clear session from Redis for a given API key"""
    redis_client.delete(f"session:{api_key}")
    redis_client.delete(f"refresh:{api_key}")
    logging.info(f"🔹 Logged out and cleared session for {api_key}")
