from dataclasses import dataclass, field

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


@deprecated("mlflow.types.llm.ChatMessage")
@dataclass
class Message:
    role: str = "user"  # "system", "user", or "assistant"
    content: str = "What is mlflow?"


@deprecated("mlflow.types.llm.ChatCompletionRequest")
@dataclass
class ChatCompletionRequest:
    messages: list[Message] = field(default_factory=lambda: [Message()])


@deprecated("mlflow.types.llm.ChatCompletionRequest")
@dataclass
class SplitChatMessagesRequest:
    query: str = "What is mlflow?"
    history: list[Message] | None = field(default_factory=list)


@deprecated("mlflow.types.llm.ChatCompletionRequest")
@dataclass
class MultiturnChatRequest:
    query: str = "What is mlflow?"
    history: list[Message] | None = field(default_factory=list)


@deprecated("mlflow.types.llm.ChatChoice")
@dataclass
class ChainCompletionChoice:
    index: int = 0
    message: Message = field(
        default_factory=lambda: Message(
            role="assistant",
            content="MLflow is an open source platform for the machine learning lifecycle.",
        )
    )
    finish_reason: str = "stop"


@deprecated("mlflow.types.llm.ChatCompletionChunk")
@dataclass
class ChainCompletionChunk:
    index: int = 0
    delta: Message = field(
        default_factory=lambda: Message(
            role="assistant",
            content="MLflow is an open source platform for the machine learning lifecycle.",
        )
    )
    finish_reason: str = "stop"


@deprecated("mlflow.types.llm.ChatCompletionResponse")
@dataclass
class ChatCompletionResponse:
    choices: list[ChainCompletionChoice] = field(default_factory=lambda: [ChainCompletionChoice()])
    object: str = "chat.completion"
    # TODO: support ChainCompletionChunk in the future


@deprecated("mlflow.types.llm.ChatCompletionResponse")
@dataclass
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
