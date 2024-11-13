from dataclasses import dataclass, field
from typing import Optional

from mlflow.models import ModelSignature
from mlflow.types.schema import (
    Array,
    ColSpec,
    DataType,
    Object,
    Property,
    Schema,
)
from mlflow.utils.annotations import deprecated


@dataclass
@deprecated("mlflow.types.llm.ChatMessage")
class Message:
    role: str = "user"  # "system", "user", or "assistant"
    content: str = "What is mlflow?"


@dataclass
@deprecated("mlflow.types.llm.ChatCompletionRequest")
class ChatCompletionRequest:
    messages: list[Message] = field(default_factory=lambda: [Message()])


@dataclass
@deprecated("mlflow.types.llm.ChatCompletionRequest")
class SplitChatMessagesRequest:
    query: str = "What is mlflow?"
    history: Optional[list[Message]] = field(default_factory=list)


@dataclass
@deprecated("mlflow.types.llm.ChatCompletionRequest")
class MultiturnChatRequest:
    query: str = "What is mlflow?"
    history: Optional[list[Message]] = field(default_factory=list)


@dataclass
@deprecated("mlflow.types.llm.ChatChoice")
class ChainCompletionChoice:
    index: int = 0
    message: Message = field(
        default_factory=lambda: Message(
            role="assistant",
            content="MLflow is an open source platform for the machine learning lifecycle.",
        )
    )
    finish_reason: str = "stop"


@dataclass
@deprecated("mlflow.types.llm.ChatCompletionChunk")
class ChainCompletionChunk:
    index: int = 0
    delta: Message = field(
        default_factory=lambda: Message(
            role="assistant",
            content="MLflow is an open source platform for the machine learning lifecycle.",
        )
    )
    finish_reason: str = "stop"


@dataclass
@deprecated("mlflow.types.llm.ChatCompletionResponse")
class ChatCompletionResponse:
    choices: list[ChainCompletionChoice] = field(default_factory=lambda: [ChainCompletionChoice()])
    object: str = "chat.completion"
    # TODO: support ChainCompletionChunk in the future


@dataclass
@deprecated("mlflow.types.llm.ChatCompletionResponse")
class StringResponse:
    content: str = "MLflow is an open source platform for the machine learning lifecycle."


CHAT_COMPLETION_REQUEST_SCHEMA = Schema(
    [
        ColSpec(
            name="messages",
            type=Array(
                Object(
                    [
                        Property("role", DataType.string),
                        Property("content", DataType.string),
                    ]
                )
            ),
        ),
    ]
)

CHAT_COMPLETION_RESPONSE_SCHEMA = Schema(
    [
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
                                ]
                            ),
                        ),
                        Property("finish_reason", DataType.string),
                    ]
                )
            ),
        ),
    ]
)

SIGNATURE_FOR_LLM_INFERENCE_TASK = {
    "llm/v1/chat": ModelSignature(
        inputs=CHAT_COMPLETION_REQUEST_SCHEMA, outputs=CHAT_COMPLETION_RESPONSE_SCHEMA
    ),
}
