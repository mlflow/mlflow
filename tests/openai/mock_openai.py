import argparse
import json
from typing import Any

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


def _make_chat_stream_chunk(content, include_usage: bool = False):
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
        }
        if include_usage
        else None,
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


async def chat_response_stream(include_usage: bool = False):
    # OpenAI Chat Completion stream only includes usage in the last chunk
    # if {"stream_options": {"include_usage": True}} is specified in the request.
    yield _make_chat_stream_chunk("Hello", include_usage=False)
    yield _make_chat_stream_chunk(" world", include_usage=include_usage)


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
            content = (
                f"data: {json.dumps(d)}\n\n"
                async for d in chat_response_stream(
                    include_usage=(payload.stream_options or {}).get("include_usage", False)
                )
            )

        return StreamingResponse(
            content,
            media_type="text/event-stream",
        )
    else:
        return chat_response(payload)


def _make_responses_payload(outputs, tools=None):
    return {
        "id": "responses-123",
        "object": "response",
        "created": 1589478378,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "max_output_tokens": None,
        "model": "gpt-4o",
        "output": outputs,
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {"effort": None, "generate_summary": None},
        "store": True,
        "temperature": 1.0,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": tools or [],
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


_DUMMY_TEXT_OUTPUTS = [
    {
        "type": "message",
        "id": "test",
        "status": "completed",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": "Dummy output",
            }
        ],
    }
]

_DUMMY_WEB_SEARCH_OUTPUTS = [
    {"type": "web_search_call", "id": "tool_call_1", "status": "completed"},
    {
        "type": "message",
        "id": "msg",
        "status": "completed",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": "As of today, March 9, 2025, one notable positive news story...",
                "annotations": [
                    {
                        "type": "url_citation",
                        "start_index": 442,
                        "end_index": 557,
                        "url": "https://.../?utm_source=chatgpt.com",
                        "title": "...",
                    },
                ],
            }
        ],
    },
]

_DUMMY_FILE_SEARCH_OUTPUTS = [
    {
        "type": "file_search_call",
        "id": "file_search_1",
        "status": "completed",
        "queries": ["attributes of an ancient brown dragon"],
        "results": None,
    },
    {
        "type": "message",
        "id": "file_search_1",
        "status": "completed",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": "The attributes of an ancient brown dragon include...",
                "annotations": [
                    {
                        "type": "file_citation",
                        "index": 320,
                        "file_id": "file-4wDz5b167pAf72nx1h9eiN",
                        "filename": "dragons.pdf",
                    },
                    {
                        "type": "file_citation",
                        "index": 576,
                        "file_id": "file-4wDz5b167pAf72nx1h9eiN",
                        "filename": "dragons.pdf",
                    },
                ],
            }
        ],
    },
]

_DUMMY_COMPUTER_USE_OUTPUTS = [
    {
        "type": "reasoning",
        "id": "rs_67cc...",
        "summary": [{"type": "summary_text", "text": "Clicking on the browser address bar."}],
    },
    {
        "type": "computer_call",
        "id": "cu_67cc...",
        "call_id": "computer_call_1",
        "action": {"type": "click", "button": "left", "x": 156, "y": 50},
        "pending_safety_checks": [],
        "status": "completed",
    },
]

_DUMMY_FUNCTION_CALL_OUTPUTS = [
    {
        "type": "function_call",
        "id": "fc_67ca09c6bedc8190a7abfec07b1a1332096610f474011cc0",
        "call_id": "function_call_1",
        "name": "get_current_weather",
        "arguments": '{"location":"Boston, MA","unit":"celsius"}',
        "status": "completed",
    }
]

_DUMMY_RESPONSES_STREAM_EVENTS = [
    {
        "type": "response.created",
        "response": _make_responses_payload(outputs=[]),
    },
    {
        "content_index": 0,
        "delta": "Hello ",
        "item_id": 0,
        "output_index": 0,
        "type": "response.output_text.delta",
    },
    {
        "content_index": 0,
        "delta": "World",
        "item_id": 0,
        "output_index": 0,
        "type": "response.output_text.delta",
    },
    {
        "response": _make_responses_payload(outputs=_DUMMY_TEXT_OUTPUTS),
        "type": "response.completed",
    },
]


class ResponsesPayload(BaseModel):
    input: Any
    tools: list[Any] | None = None
    stream: bool = False


@app.post("/responses", response_model_exclude_unset=True)
async def responses(payload: ResponsesPayload):
    if payload.stream:
        content = (
            f"event: {d['type']}\ndata: {json.dumps(d)}\n\n" for d in _DUMMY_RESPONSES_STREAM_EVENTS
        )
        return StreamingResponse(content, media_type="text/event-stream")

    if tools := payload.tools or []:
        if tools[0]["type"] == "web_search_preview":
            outputs = _DUMMY_WEB_SEARCH_OUTPUTS
        elif tools[0]["type"] == "file_search":
            outputs = _DUMMY_FILE_SEARCH_OUTPUTS
        elif tools[0]["type"] == "computer_use_preview":
            outputs = _DUMMY_COMPUTER_USE_OUTPUTS
        elif tools[0]["type"] == "function":
            outputs = _DUMMY_FUNCTION_CALL_OUTPUTS
        else:
            raise fastapi.HTTPException(
                status_code=400,
                detail=f"Unsupported tool type: {tools[0]['type']}",
            )
        return _make_responses_payload(outputs, tools)

    return _make_responses_payload(outputs=_DUMMY_TEXT_OUTPUTS)


class CompletionsPayload(BaseModel):
    prompt: str | list[str]
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
    input: str | list[str]


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


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str)
    parser.add_argument("--port", type=int)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
