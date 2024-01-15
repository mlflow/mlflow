from typing import List, Optional

from pydantic import BaseModel

from mlflow.types.schema import Array, ColSpec, DataType, Object, Property, Schema


class ChatMessage(BaseModel):
    """
    A pydantic model representing a single message in a chat conversation.
    """

    role: str
    content: str
    name: Optional[str] = None


class ChatParams(BaseModel):
    """
    A pydantic model representing a standard set of parameters for chat model inference.
    """

    temperature: float = 1.0
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    n: int = 1
    stream: bool = False


class ChatChoice(BaseModel):
    """
    A pydantic model representing a single choice returned by the model in a chat response.
    """

    index: int
    message: ChatMessage
    finish_reason: str


class ChatRequest(ChatParams):
    """
    A pydantic model representing the request body for a chat model inference request.
    """

    messages: List[ChatMessage]


class TokenUsageStats(BaseModel):
    """
    A pydantic model representing token usage statistics for a chat model inference request.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    """
    A pydantic model representing the response body for a chat model inference request.
    """

    id: str
    object: str
    created: int
    model: str
    choices: List[ChatChoice]
    usage: TokenUsageStats


CHAT_MODEL_INPUT_SCHEMA = Schema(
    [
        ColSpec(
            name="messages",
            type=Array(
                Object(
                    [
                        Property("role", DataType.string),
                        Property("content", DataType.string),
                        Property("name", DataType.string, False),
                    ]
                )
            ),
        ),
        ColSpec(name="temperature", type=DataType.double, required=False),
        ColSpec(name="max_tokens", type=DataType.long, required=False),
        ColSpec(name="stop", type=Array(DataType.string), required=False),
        ColSpec(name="n", type=DataType.long, required=False),
        ColSpec(name="stream", type=DataType.boolean, required=False),
    ]
)

CHAT_MODEL_OUTPUT_SCHEMA = Schema(
    [
        ColSpec(name="id", type=DataType.string),
        ColSpec(name="object", type=DataType.string),
        ColSpec(name="created", type=DataType.long),
        ColSpec(name="model", type=DataType.string),
        ColSpec(
            name="choices",
            type=Array(
                Object(
                    [
                        Property("index", DataType.long),
                        Property(
                            "message",
                            Object(
                                [
                                    Property("role", DataType.string),
                                    Property("content", DataType.string),
                                    Property("name", DataType.string, False),
                                ]
                            ),
                        ),
                        Property("finish_reason", DataType.string),
                    ]
                )
            ),
        ),
        ColSpec(
            name="usage",
            type=Object(
                [
                    Property("prompt_tokens", DataType.long),
                    Property("completion_tokens", DataType.long),
                    Property("total_tokens", DataType.long),
                ]
            ),
        ),
    ]
)

CHAT_MODEL_INPUT_EXAMPLE = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
    "temperature": 1.0,
    "max_tokens": 20,
    "stop": ["\n"],
    "n": 1,
    "stream": False,
}
