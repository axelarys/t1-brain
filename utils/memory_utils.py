# utils/memory_utils.py

import hashlib
from config.settings import EMBEDDING_API_KEYS

def get_api_key(memory_type: str) -> str:
    """
    Get the correct API key based on memory type (text or image).
    Defaults to text key.
    """
    return EMBEDDING_API_KEYS.get(memory_type, EMBEDDING_API_KEYS["text"])

def get_sha256_hash(data: str) -> str:
    """
    Generate SHA-256 hash from string data (text or image URL).
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()
