import httpx

# URL of your running FastAPI app
API_URL = "http://localhost:8000/memory/store_image"

# Test payload
payload = {
    "session_id": "test_001",
    "image_url": "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png",
    "response": "User uploaded a reference image for demo",
    "sentiment": "neutral"
}

headers = {
    "Content-Type": "application/json",
    "X-API-KEY": "WsRocks1234"  # replace if your system uses a different key
}

# Send POST request
response = httpx.post(API_URL, json=payload, headers=headers)

# Show result
print("Status Code:", response.status_code)
print("Response:", response.json())
