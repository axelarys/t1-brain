import httpx

API_URL = "http://localhost:8000/memory/store_image"
payload = {
    "session_id": "test_001",
    "image_url": "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png",
    "response": "Test image response",
    "sentiment": "neutral"
}
headers = {"Content-Type": "application/json"}

try:
    response = httpx.post(API_URL, json=payload, headers=headers, timeout=60)
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())
except Exception as e:
    print("❌ Request failed:", str(e))
