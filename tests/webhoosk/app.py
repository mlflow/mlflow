import json
from pathlib import Path

import fastapi
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
    with LOG_FILE.open("ab") as f:
        f.write(json.dumps(webhook_data).encode("utf-8"))
        f.write(b"\n")

    return {"status": "received"}


@app.delete("/logs")
async def clear_logs():
    if LOG_FILE.exists():
        LOG_FILE.unlink()
    return {"status": "logs cleared"}


@app.get("/logs")
async def get_logs():
    if not LOG_FILE.exists():
        return {"logs": []}

    logs = []
    with LOG_FILE.open("r") as f:
        for line in f:
            if line.strip():
                logs.append(json.loads(line.strip()))

    return {"logs": logs}
