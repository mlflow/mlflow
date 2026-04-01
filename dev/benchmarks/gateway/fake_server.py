# /// script
# requires-python = ">=3.10"
# dependencies = ["fastapi", "uvicorn[standard]"]
# ///
"""Fake OpenAI-compatible server for benchmarking.

Returns synthetic responses after a configurable delay so benchmarks measure
MLflow overhead rather than provider latency.

Run standalone:
    uv run fake_server.py

Or via gunicorn (as launched by run.py):
    gunicorn -k uvicorn.workers.UvicornWorker -w 8 -b 0.0.0.0:9000 fake_server:app
"""

import asyncio
import os
import time

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

DELAY_MS = int(os.environ.get("FAKE_RESPONSE_DELAY_MS", "50"))


class ChatRequest(BaseModel):
    model: str = "gpt-3.5-turbo"
    messages: list = []
    stream: bool = False
    temperature: float = 1.0
    max_tokens: int = 50


class CompletionsRequest(BaseModel):
    model: str = "gpt-3.5-turbo"
    prompt: str = ""
    max_tokens: int = 50


class EmbeddingsRequest(BaseModel):
    model: str = "text-embedding-ada-002"
    input: str | list = ""


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    await asyncio.sleep(DELAY_MS / 1000)
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


@app.post("/v1/completions")
async def completions(req: CompletionsRequest):
    await asyncio.sleep(DELAY_MS / 1000)
    return {
        "id": "cmpl-fake",
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"text": "Hello!", "index": 0, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


@app.post("/v1/embeddings")
async def embeddings(req: EmbeddingsRequest):
    await asyncio.sleep(DELAY_MS / 1000)
    return {
        "object": "list",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.001] * 1536}],
        "model": "text-embedding-ada-002",
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("fake_server:app", host="0.0.0.0", port=9137, log_level="warning")
