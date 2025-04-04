import json
from typing import Any, Union

import fastapi
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from mlflow.types.chat import ChatCompletionRequest
from mlflow.utils import IS_PYDANTIC_V2_OR_NEWER

EMPTY_CHOICES = "EMPTY_CHOICES"

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


async def chat_response_stream():
    yield _make_chat_stream_chunk("Hello")
    yield _make_chat_stream_chunk(" world")


async def chat_response_stream_empty_choices():
    yield _make_chat_stream_chunk_empty_choices()
    yield _make_chat_stream_chunk("Hello")


@app.post("/chat/completions", response_model_exclude_unset=True)
async def chat(payload: ChatCompletionRequest):
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


class ResponsesPayload(BaseModel):
    # Text or one of supported content types: https://platform.openai.com/docs/api-reference/responses/create
    input: Union[str, list[Any]]


@app.post("/responses")
def responses(payload: ResponsesPayload):
    return {
        "id": "responses-123",
        "object": "response",
        "created": 1589478378,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "max_output_tokens": None,
        "model": "gpt-4o-mini",
        "output": [
            {
                "type": "message",
                "id": "msg_67ccd2bf17f0819081ff3bb2cf6508e60bb6a6b452d3795b",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Hello world",
                    }
                ],
            }
        ],
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {"effort": None, "generate_summary": None},
        "store": True,
        "temperature": 1.0,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 36,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 87,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 123,
        },
        "user": None,
        "metadata": {},
    }
