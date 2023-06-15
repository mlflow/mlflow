"""
An app that mimics the OpenAI API for testing purposes. To use this app, run the following commands:
```
export OPENAI_API_KEY="sk-fake"
sed -i '/openai_api_base: /s|https://api.openai.com/v1|http://127.0.0.1:8000|g' examples/gateway/config.yaml
uvicorn examples.gateway.fake_openai:app --reload
```
"""

import random
from fastapi import FastAPI, Request

app = FastAPI()


@app.post("/chat/completions")
async def chat(request: Request):
    print(request.json())
    return {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-3.5-turbo-0301",
        "usage": {
            "prompt_tokens": 13,
            "completion_tokens": 7,
            "total_tokens": 20,
        },
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "\n\nThis is a test!",
                },
                "finish_reason": "stop",
                "index": 0,
            }
        ],
    }


@app.post("/completions")
async def completions(request: Request):
    print(request.json())
    return {
        "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
        "object": "text_completion",
        "created": 1589478378,
        "model": "text-davinci-003",
        "choices": [
            {
                "text": "\n\nThis is indeed a test",
                "index": 0,
                "logprobs": None,
                "finish_reason": "length",
            }
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 7,
            "total_tokens": 12,
        },
    }


@app.post("/embeddings")
async def embeddings(request: Request):
    print(request.json())
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [random.random() for _ in range(10)],
                "index": 0,
            }
        ],
        "model": "text-embedding-ada-002",
        "usage": {"prompt_tokens": 8, "total_tokens": 8},
    }
