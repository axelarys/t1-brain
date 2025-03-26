from fastapi import FastAPI, Header, HTTPException
from google.oauth2 import id_token
from google.auth.transport import requests

app = FastAPI()

def validate_google_oauth(token: str):
    try:
        info = id_token.verify_oauth2_token(token, requests.Request())
        if info["email"] != "t1brain-auth@t1-brain.iam.gserviceaccount.com":
            raise HTTPException(status_code=403, detail="Unauthorized email")
        return info
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/validate-token")
async def validate_token(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=400, detail="Authorization header missing")
    
    token = authorization.replace("Bearer ", "")
    user_info = validate_google_oauth(token)
    return {"message": "Token is valid", "user": user_info["email"]}
