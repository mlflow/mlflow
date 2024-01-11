from typing import List, Optional

from pydantic import BaseModel

from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Array, ColSpec, DataType, Object, Property, Schema


class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class ChatParams(BaseModel):
    temperature: float = 1.0
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    n: int = 1
    stream: bool = False


class LogProbCandidate(BaseModel):
    token: str
    logprob: float
    bytes: List[int]


# represents the most likely token, along with a list of the next top tokens
class LogProbEntry(BaseModel):
    token: str
    logprob: float
    bytes: List[int]
    top_logprobs: List[LogProbCandidate]


# represents the value of the "logprobs" key in the response
class LogProbsValue(BaseModel):
    content: List[LogProbEntry]


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    logprobs: Optional[LogProbsValue]
    finish_reason: str


class ChatRequest(ChatParams):
    messages: List[ChatMessage]


class ChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    system_fingerprint: Optional[str]
    choices: List[ChatChoice]
    usage: dict


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
        ColSpec(name="system_fingerprint", type=DataType.string, required=False),
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
                        Property(
                            "logprobs",
                            Object(
                                [
                                    Property(
                                        "content",
                                        Array(
                                            Object(
                                                [
                                                    Property("token", DataType.string),
                                                    Property("logprob", DataType.double),
                                                    Property("bytes", Array(DataType.long)),
                                                    Property(
                                                        "top_logprobs",
                                                        Array(
                                                            Object(
                                                                [
                                                                    Property(
                                                                        "token", DataType.string
                                                                    ),
                                                                    Property(
                                                                        "logprob", DataType.double
                                                                    ),
                                                                    Property(
                                                                        "bytes",
                                                                        Array(DataType.long),
                                                                    ),
                                                                ]
                                                            )
                                                        ),
                                                    ),
                                                ]
                                            )
                                        ),
                                    )
                                ]
                            ),
                            False,
                        ),
                        Property("finish_reason", DataType.string),
                    ]
                )
            ),
        ),
    ]
)

CHAT_MODEL_SIGNATURE = ModelSignature(
    inputs=CHAT_MODEL_INPUT_SCHEMA,
    outputs=CHAT_MODEL_OUTPUT_SCHEMA,
)

CHAT_MODEL_INPUT_EXAMPLE = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
    "temperature": 1.0,
    "max_tokens": 20,
    "stop": None,
    "n": 1,
    "stream": False,
}
