import json
from typing import List, Union

import fastapi
from pydantic import BaseModel

app = fastapi.FastAPI()


@app.get("/health")
def health():
    return {"status": "healthy"}


class Message(BaseModel):
    role: str
    content: str


class ChatPayload(BaseModel):
    messages: List[Message]


@app.post("/chat/completions")
async def chat(payload: ChatPayload):
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-3.5-turbo-0613",
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


class CompletionsPayload(BaseModel):
    prompt: Union[str, List[str]]


@app.post("/completions")
def completions(payload: CompletionsPayload):
    return {
        "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
        "object": "text_completion",
        "created": 1589478378,
        "model": "gpt-3.5-turbo",
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
