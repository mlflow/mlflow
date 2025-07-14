import json
import sys
from pathlib import Path

import fastapi
import uvicorn
from fastapi import Request

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


if __name__ == "__main__":
    port = sys.argv[1]
    uvicorn.run(app, host="0.0.0.0", port=int(port))
