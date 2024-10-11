import json
from typing import List, Union

import fastapi
from pydantic import BaseModel
from starlette.responses import StreamingResponse

EMPTY_CHOICES = "EMPTY_CHOICES"

app = fastapi.FastAPI()


@app.get("/health")
def health():
    return {"status": "healthy"}


class Message(BaseModel):
    role: str
    content: str


class ChatPayload(BaseModel):
    messages: List[Message]
    temperature: float = 0
    stream: bool = False


def chat_response(payload: ChatPayload):
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4o-mini",
        "system_fingerprint": "fp_44709d6fcb",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps([m.dict() for m in payload.messages]),
                },
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21,
        },
    }


def _make_chat_stream_chunk(content):
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1677652288,
        "model": "gpt-4o-mini",
        "system_fingerprint": "fp_44709d6fcb",
        "choices": [
            {
                "delta": {
                    "content": content,
                    "function_call": None,
                    "role": None,
                    "tool_calls": None,
                },
                "finish_reason": None,
                "index": 0,
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21,
        },
    }


def _make_chat_stream_chunk_empty_choices():
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1677652288,
        "model": "gpt-4o-mini",
        "system_fingerprint": "fp_44709d6fcb",
        "choices": [],
        "usage": None,
    }


async def chat_response_stream():
    yield _make_chat_stream_chunk("Hello")
    yield _make_chat_stream_chunk(" world")


async def chat_response_stream_empty_choices():
    yield _make_chat_stream_chunk_empty_choices()
    yield _make_chat_stream_chunk("Hello")


@app.post("/chat/completions")
async def chat(payload: ChatPayload):
    if not 0.0 <= payload.temperature <= 2.0:
        return fastapi.Response(
            content="Temperature must be between 0.0 and 2.0",
            status_code=400,
        )
    if payload.stream:
        # SSE stream
        if EMPTY_CHOICES == payload.messages[0].content:
            content = (
                f"data: {json.dumps(d)}\n\n" async for d in chat_response_stream_empty_choices()
            )
        else:
            content = (f"data: {json.dumps(d)}\n\n" async for d in chat_response_stream())

        return StreamingResponse(
            content,
            media_type="text/event-stream",
        )
    else:
        return chat_response(payload)


class CompletionsPayload(BaseModel):
    prompt: Union[str, List[str]]
    stream: bool = False


def completions_response(payload: CompletionsPayload):
    return {
        "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
        "object": "text_completion",
        "created": 1589478378,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "text": text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "length",
            }
            for text in ([payload.prompt] if isinstance(payload.prompt, str) else payload.prompt)
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }


def _make_completions_stream_chunk(content):
    return {
        "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
        "object": "text_completion",
        "created": 1589478378,
        "model": "gpt-4o-mini",
        "choices": [{"finish_reason": None, "index": 0, "logprobs": None, "text": content}],
        "system_fingerprint": None,
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }


def _make_completions_stream_chunk_empty_choices():
    return {
        "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
        "object": "text_completion",
        "created": 1589478378,
        "model": "gpt-4o-mini",
        "choices": [],
        "system_fingerprint": None,
        "usage": None,
    }


async def completions_response_stream():
    yield _make_completions_stream_chunk("Hello")
    yield _make_completions_stream_chunk(" world")


async def completions_response_stream_empty_choices():
    yield _make_completions_stream_chunk_empty_choices()
    yield _make_completions_stream_chunk("Hello")


@app.post("/completions")
def completions(payload: CompletionsPayload):
    if payload.stream:
        if EMPTY_CHOICES == payload.prompt:
            content = (
                f"data: {json.dumps(d)}\n\n"
                async for d in completions_response_stream_empty_choices()
            )
        else:
            content = (f"data: {json.dumps(d)}\n\n" async for d in completions_response_stream())

        return StreamingResponse(
            content,
            media_type="text/event-stream",
        )
    else:
        return completions_response(payload)


class EmbeddingsPayload(BaseModel):
    input: Union[str, List[str]]


@app.post("/embeddings")
def embeddings(payload: EmbeddingsPayload):
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": list(range(1536)),
                "index": 0,
            }
            for _ in range(1 if isinstance(payload.input, str) else len(payload.input))
        ],
        "model": "text-embedding-ada-002",
        "usage": {"prompt_tokens": 8, "total_tokens": 8},
    }


@app.get("/models/{model}")
def models(model: str):
    return {
        "id": model,
        "object": "model",
        "created": 1686935002,
        "owned_by": "openai",
    }
