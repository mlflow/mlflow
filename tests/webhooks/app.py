import base64
import hashlib
import hmac
import itertools
import json
import sys
from pathlib import Path

import fastapi
import uvicorn
from fastapi import HTTPException, Request

from mlflow.webhooks.constants import (
    WEBHOOK_DELIVERY_ID_HEADER,
    WEBHOOK_SIGNATURE_HEADER,
    WEBHOOK_SIGNATURE_VERSION,
    WEBHOOK_TIMESTAMP_HEADER,
)

LOG_FILE = Path("logs.jsonl")

app = fastapi.FastAPI()


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/insecure-webhook")
async def insecure_webhook(request: Request):
    payload = await request.json()
    # Extract the data field from webhook payload
    actual_payload = payload.get("data", payload)
    webhook_data = {
        "endpoint": "/insecure-webhook",
        "payload": actual_payload,
        "headers": dict(request.headers),
        "status_code": 200,
        "error": None,
    }
    with LOG_FILE.open("a") as f:
        f.write(json.dumps(webhook_data) + "\n")

    return {"status": "received"}


@app.post("/reset")
async def reset():
    """Reset both logs and counters for testing"""
    global flaky_counter, rate_limited_counter

    # Clear logs
    if LOG_FILE.exists():
        LOG_FILE.open("w").close()

    # Reset all counters
    flaky_counter = itertools.count(1)
    rate_limited_counter = itertools.count(1)

    return {"status": "reset complete", "logs": "cleared", "counters": "reset"}


@app.get("/logs")
async def get_logs():
    if not LOG_FILE.exists():
        return {"logs": []}

    with LOG_FILE.open("r") as f:
        logs = [json.loads(s) for line in f if (s := line.strip())]
        return {"logs": logs}


# Secret key for HMAC verification (in real world, this would be stored securely)
WEBHOOK_SECRET = "test-secret-key"


def verify_webhook_signature(
    payload: str, signature: str, delivery_id: str, timestamp: str
) -> bool:
    if not signature or not signature.startswith(f"{WEBHOOK_SIGNATURE_VERSION},"):
        return False

    # Signature format: delivery_id.timestamp.payload
    signed_content = f"{delivery_id}.{timestamp}.{payload}"
    expected_signature = hmac.new(
        WEBHOOK_SECRET.encode("utf-8"), signed_content.encode("utf-8"), hashlib.sha256
    ).digest()
    expected_signature_b64 = base64.b64encode(expected_signature).decode("utf-8")

    provided_signature = signature.removeprefix(f"{WEBHOOK_SIGNATURE_VERSION},")
    return hmac.compare_digest(expected_signature_b64, provided_signature)


@app.post("/secure-webhook")
async def secure_webhook(request: Request):
    body = await request.body()
    signature = request.headers.get(WEBHOOK_SIGNATURE_HEADER)
    timestamp = request.headers.get(WEBHOOK_TIMESTAMP_HEADER)
    delivery_id = request.headers.get(WEBHOOK_DELIVERY_ID_HEADER)

    if not signature:
        error_data = {
            "endpoint": "/secure-webhook",
            "headers": dict(request.headers),
            "status_code": 400,
            "payload": None,
            "error": "Missing signature header",
        }
        with LOG_FILE.open("a") as f:
            f.write(json.dumps(error_data) + "\n")
        raise HTTPException(status_code=400, detail="Missing signature header")

    if not timestamp:
        error_data = {
            "endpoint": "/secure-webhook",
            "error": "Missing timestamp header",
            "status_code": 400,
            "headers": dict(request.headers),
        }
        with LOG_FILE.open("a") as f:
            f.write(json.dumps(error_data) + "\n")
        raise HTTPException(status_code=400, detail="Missing timestamp header")

    if not delivery_id:
        error_data = {
            "endpoint": "/secure-webhook",
            "error": "Missing delivery ID header",
            "status_code": 400,
            "headers": dict(request.headers),
        }
        with LOG_FILE.open("a") as f:
            f.write(json.dumps(error_data) + "\n")
        raise HTTPException(status_code=400, detail="Missing delivery ID header")

    if not verify_webhook_signature(body.decode("utf-8"), signature, delivery_id, timestamp):
        error_data = {
            "endpoint": "/secure-webhook",
            "headers": dict(request.headers),
            "status_code": 401,
            "payload": None,
            "error": "Invalid signature",
        }
        with LOG_FILE.open("a") as f:
            f.write(json.dumps(error_data) + "\n")
        raise HTTPException(status_code=401, detail="Invalid signature")

    payload = json.loads(body)
    # Extract the data field from webhook payload
    actual_payload = payload.get("data", payload)
    webhook_data = {
        "endpoint": "/secure-webhook",
        "payload": actual_payload,
        "headers": dict(request.headers),
        "status_code": 200,
        "error": None,
    }

    with LOG_FILE.open("a") as f:
        f.write(json.dumps(webhook_data) + "\n")

    return {"status": "received", "signature": "verified"}


# Create separate counters for each endpoint using itertools.count
flaky_counter = itertools.count(1)
rate_limited_counter = itertools.count(1)


@app.post("/flaky-webhook")
async def flaky_webhook(request: Request):
    """Endpoint that fails initially but succeeds after retries"""
    attempt = next(flaky_counter)

    payload = await request.json()
    actual_payload = payload.get("data", payload)

    # Log the attempt
    webhook_data = {
        "endpoint": "/flaky-webhook",
        "payload": actual_payload,
        "headers": dict(request.headers),
        "attempt": attempt,
        "error": None,
    }

    # Fail on first two attempts with 500 error
    if attempt <= 2:
        webhook_data["status_code"] = 500
        webhook_data["error"] = "Server error (will retry)"
        with LOG_FILE.open("a") as f:
            f.write(json.dumps(webhook_data) + "\n")
        raise HTTPException(status_code=500, detail="Internal server error")

    # Succeed on third attempt
    webhook_data["status_code"] = 200
    with LOG_FILE.open("a") as f:
        f.write(json.dumps(webhook_data) + "\n")

    return {"status": "received", "attempt": attempt}


@app.post("/rate-limited-webhook")
async def rate_limited_webhook(request: Request):
    """Endpoint that returns 429 with Retry-After header"""
    attempt = next(rate_limited_counter)

    payload = await request.json()
    actual_payload = payload.get("data", payload)

    # Log the attempt
    webhook_data = {
        "endpoint": "/rate-limited-webhook",
        "payload": actual_payload,
        "headers": dict(request.headers),
        "attempt": attempt,
        "error": None,
    }

    # Return 429 on first attempt
    if attempt == 1:
        webhook_data["status_code"] = 429
        webhook_data["error"] = "Rate limited"
        with LOG_FILE.open("a") as f:
            f.write(json.dumps(webhook_data) + "\n")
        # Return 429 with Retry-After header
        response = fastapi.Response(content="Rate limited", status_code=429)
        response.headers["Retry-After"] = "2"
        return response

    # Succeed on second attempt
    webhook_data["status_code"] = 200
    with LOG_FILE.open("a") as f:
        f.write(json.dumps(webhook_data) + "\n")

    return {"status": "received", "attempt": attempt}


if __name__ == "__main__":
    port = sys.argv[1]
    uvicorn.run(app, host="0.0.0.0", port=int(port))
