import hashlib
import hmac
import json
import sys
from pathlib import Path

import fastapi
import uvicorn
from fastapi import HTTPException, Request

LOG_FILE = Path("logs.jsonl")

app = fastapi.FastAPI()


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/insecure-webhook")
async def insecure_webhook(request: Request):
    webhook_data = {
        "endpoint": "/insecure-webhook",
        "payload": await request.json(),
        "headers": dict(request.headers),
        "status_code": 200,
    }
    with LOG_FILE.open("a") as f:
        f.write(json.dumps(webhook_data) + "\n")

    return {"status": "received"}


@app.delete("/logs")
async def clear_logs():
    if LOG_FILE.exists():
        # Clear contents
        LOG_FILE.open("w").close()
    return {"status": "logs cleared"}


@app.get("/logs")
async def get_logs():
    if not LOG_FILE.exists():
        return {"logs": []}

    with LOG_FILE.open("r") as f:
        logs = [json.loads(s) for line in f if (s := line.strip())]
        return {"logs": logs}


# Secret key for HMAC verification (in real world, this would be stored securely)
WEBHOOK_SECRET = "test-secret-key"


def verify_signature(payload: bytes, signature: str) -> bool:
    if not signature or not signature.startswith("sha256="):
        return False

    expected_signature = hmac.new(
        WEBHOOK_SECRET.encode("utf-8"), payload, hashlib.sha256
    ).hexdigest()
    provided_digest = signature.removeprefix("sha256=")
    return hmac.compare_digest(expected_signature, provided_digest)


@app.post("/secure-webhook")
async def secure_webhook(request: Request):
    body = await request.body()
    signature = request.headers.get("X-MLflow-Signature")

    if not signature:
        error_data = {
            "endpoint": "/secure-webhook",
            "error": "Missing signature header",
            "status_code": 400,
            "headers": dict(request.headers),
        }
        with LOG_FILE.open("a") as f:
            f.write(json.dumps(error_data) + "\n")
        raise HTTPException(status_code=400, detail="Missing signature header")

    if not verify_signature(body, signature):
        error_data = {
            "endpoint": "/secure-webhook",
            "error": "Invalid signature",
            "status_code": 401,
            "headers": dict(request.headers),
        }
        with LOG_FILE.open("a") as f:
            f.write(json.dumps(error_data) + "\n")
        raise HTTPException(status_code=401, detail="Invalid signature")

    payload = json.loads(body)
    webhook_data = {
        "endpoint": "/secure-webhook",
        "payload": payload,
        "headers": dict(request.headers),
        "status_code": 200,
    }

    with LOG_FILE.open("a") as f:
        f.write(json.dumps(webhook_data) + "\n")

    return {"status": "received", "signature": "verified"}


if __name__ == "__main__":
    port = sys.argv[1]
    uvicorn.run(app, host="0.0.0.0", port=int(port))
