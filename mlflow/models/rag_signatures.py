from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Message:
    role: str = "user"  # "system", "user", or "assistant"
    content: str = "What is mlflow?"


@dataclass
class ChatCompletionRequest:
    messages: List[Message] = field(default_factory=lambda: [Message()])


@dataclass
class MultiturnChatRequest:
    query: str = "What is mlflow?"
    history: Optional[List[Message]] = field(default_factory=list)


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


@dataclass
class ChatCompletionResponse:
    choices: List[ChainCompletionChoice] = field(default_factory=lambda: [ChainCompletionChoice()])
    # TODO: support ChainCompletionChunk in the future


@dataclass
class StringResponse:
    content: str = "MLflow is an open source platform for the machine learning lifecycle."
