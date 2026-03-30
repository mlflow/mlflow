"""
Fake OpenAI-compatible server for benchmarking the MLflow AI Gateway.

Returns static JSON responses matching the shapes the OpenAI provider adapter expects.
Configurable simulated latency via FAKE_RESPONSE_DELAY_MS env var (default 50ms).
"""

import asyncio
import os
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

FAKE_RESPONSE_DELAY_MS = int(os.environ.get("FAKE_RESPONSE_DELAY_MS", "50"))

app = FastAPI(title="Fake OpenAI Server")


async def _simulate_delay():
    if FAKE_RESPONSE_DELAY_MS > 0:
        await asyncio.sleep(FAKE_RESPONSE_DELAY_MS / 1000.0)


class ChatRequest(BaseModel):
    model: str = "gpt-4o-mini"
    messages: list[dict[str, str]] = []
    temperature: float = 1.0
    max_tokens: int | None = None
    stream: bool = False


class CompletionsRequest(BaseModel):
    model: str = "gpt-4o-mini"
    prompt: str = ""
    temperature: float = 1.0
    max_tokens: int | None = None
    stream: bool = False


class EmbeddingsRequest(BaseModel):
    model: str = "text-embedding-ada-002"
    input: str | list[str] = ""


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    await _simulate_delay()
    return JSONResponse(
        content={
            "id": "chatcmpl-benchmark-fake",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "This is a fake response for benchmarking.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18,
            },
        },
        media_type="application/json",
    )


@app.post("/v1/completions")
async def completions(request: CompletionsRequest):
    await _simulate_delay()
    return JSONResponse(
        content={
            "id": "cmpl-benchmark-fake",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "text": "This is a fake completion for benchmarking.",
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 8,
                "total_tokens": 13,
            },
        },
        media_type="application/json",
    )


@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingsRequest):
    await _simulate_delay()
    # Return a 1536-dimensional fake embedding (text-embedding-ada-002 size)
    fake_embedding = [0.001] * 1536
    data = []
    inputs = request.input if isinstance(request.input, list) else [request.input]
    for i, _ in enumerate(inputs):
        data.append(
            {
                "object": "embedding",
                "embedding": fake_embedding,
                "index": i,
            }
        )
    return JSONResponse(
        content={
            "object": "list",
            "data": data,
            "model": request.model,
            "usage": {
                "prompt_tokens": 8,
                "total_tokens": 8,
            },
        },
        media_type="application/json",
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
