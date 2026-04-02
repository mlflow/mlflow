# /// script
# requires-python = ">=3.10"
# dependencies = ["fastapi>=0.115.0,<1", "uvicorn[standard]>=0.30.0,<1"]
# ///
"""Fake OpenAI-compatible server for benchmarking.

Returns synthetic responses after a configurable delay so benchmarks measure
MLflow overhead rather than provider latency.

Run standalone:
    uv run fake_server.py
    PORT=9200 uv run fake_server.py

Or with multiple workers (as launched by run.py):
    uvicorn fake_server:app --workers 8 --port 9137
"""

import asyncio
import os
import time
from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()

DELAY_MS = int(os.environ.get("FAKE_RESPONSE_DELAY_MS", "50"))


class ChatRequest(BaseModel):
    model: str = "gpt-4o-mini"
    messages: list[dict[str, str]] = Field(min_length=1)
    stream: bool = False
    temperature: float = 1.0
    max_tokens: int = 50


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest) -> dict[str, Any]:
    await asyncio.sleep(DELAY_MS / 1000)
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


@app.get("/health")
async def health() -> dict[str, str]:
    # Polled by run.py's _wait_for_port to detect when the server is ready.
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "9137"))
    host = os.environ.get("FAKE_SERVER_HOST", "127.0.0.1")
    uvicorn.run("fake_server:app", host=host, port=port, log_level="warning")
