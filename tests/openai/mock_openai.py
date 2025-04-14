import json
from enum import Enum
from typing import Union

import fastapi
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from mlflow.types.chat import ChatCompletionRequest
from mlflow.utils import IS_PYDANTIC_V2_OR_NEWER


class ChatChunkVariant(str, Enum):
    EMPTY_CHOICES = "EMPTY_CHOICES"
    CHOICE_DELTA_NONE = "CHOICE_DELTA_NONE"
    CHOICE_DELTA_CONTENT_NONE = "CHOICE_DELTA_CONTENT_NONE"


class CompletionsChunkVariant(str, Enum):
    EMPTY_CHOICES = "EMPTY_CHOICES"
    CHOICE_EMPTY_TEXT = "CHOICE_EMPTY_TEXT"


app = fastapi.FastAPI()


@app.get("/health")
def health():
    return {"status": "healthy"}


def chat_response(payload: ChatCompletionRequest):
    if IS_PYDANTIC_V2_OR_NEWER:
        dumped_input = json.dumps([m.model_dump(exclude_unset=True) for m in payload.messages])
    else:
        dumped_input = json.dumps([m.dict(exclude_unset=True) for m in payload.messages])
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
                    "content": dumped_input,
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


def _make_chat_stream_chunk_choice_delta_none():
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1677652288,
        "model": "gpt-4o-mini",
        "system_fingerprint": "fp_44709d6fcb",
        "choices": [{}],
        "usage": None,
    }


def _make_chat_stream_chunk_choice_delta_content_none():
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1677652288,
        "model": "gpt-4o-mini",
        "system_fingerprint": "fp_44709d6fcb",
        "choices": [{"delta": {}}],
        "usage": None,
    }


async def chat_response_stream():
    yield _make_chat_stream_chunk("Hello")
    yield _make_chat_stream_chunk(" world")


async def chat_response_stream_empty_choices():
    yield _make_chat_stream_chunk_empty_choices()
    yield _make_chat_stream_chunk("Hello")


async def chat_response_stream_choice_delta_none():
    yield _make_chat_stream_chunk("Hello")
    yield _make_chat_stream_chunk_choice_delta_none()


async def chat_response_stream_choice_delta_content_none():
    yield _make_chat_stream_chunk("Hello")
    yield _make_chat_stream_chunk_choice_delta_content_none()


@app.post("/chat/completions", response_model_exclude_unset=True)
async def chat(payload: ChatCompletionRequest):
    if payload.stream:
        # SSE stream
        if ChatChunkVariant.EMPTY_CHOICES == payload.messages[0].content:
            content = (
                f"data: {json.dumps(d)}\n\n" async for d in chat_response_stream_empty_choices()
            )
        elif ChatChunkVariant.CHOICE_DELTA_NONE == payload.messages[0].content:
            content = (
                f"data: {json.dumps(d)}\n\n" async for d in chat_response_stream_choice_delta_none()
            )
        elif ChatChunkVariant.CHOICE_DELTA_CONTENT_NONE == payload.messages[0].content:
            content = (
                f"data: {json.dumps(d)}\n\n"
                async for d in chat_response_stream_choice_delta_content_none()
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
    prompt: Union[str, list[str]]
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


def _make_completions_stream_chunk_choice_empty_text():
    return {
        "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
        "object": "text_completion",
        "created": 1589478378,
        "model": "gpt-4o-mini",
        "choices": [{"text": ""}],
        "system_fingerprint": None,
        "usage": None,
    }


async def completions_response_stream():
    yield _make_completions_stream_chunk("Hello")
    yield _make_completions_stream_chunk(" world")


async def completions_response_stream_empty_choices():
    yield _make_completions_stream_chunk_empty_choices()
    yield _make_completions_stream_chunk("Hello")


async def completions_response_stream_choice_empty_text():
    yield _make_completions_stream_chunk("Hello")
    yield _make_completions_stream_chunk_choice_empty_text()


@app.post("/completions")
def completions(payload: CompletionsPayload):
    if payload.stream:
        if CompletionsChunkVariant.EMPTY_CHOICES == payload.prompt:
            content = (
                f"data: {json.dumps(d)}\n\n"
                async for d in completions_response_stream_empty_choices()
            )
        elif CompletionsChunkVariant.CHOICE_EMPTY_TEXT == payload.prompt:
            content = (
                f"data: {json.dumps(d)}\n\n"
                async for d in completions_response_stream_choice_empty_text()
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
    input: Union[str, list[str]]


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
