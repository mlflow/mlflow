from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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

from langchain_core.output_parsers.transform import BaseTransformOutputParser

class ChatCompletionsOutputParser(BaseTransformOutputParser[Dict[str, Any]]):
    """OutputParser that wraps the string output into an OpenAI-like structured format."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this class is serializable."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "output_parser"]

    @property
    def _type(self) -> str:
        """Return the output parser type for serialization."""
        return "openai_style"

    def parse(self, text: str) -> Dict[str, Any]:
        """Returns the input text wrapped in an OpenAI-like response structure."""
        return {
            "choices": [
                {
                    "message": {
                        "content": text
                    }
                }
            ]
        }


class StrObjOutputParser(BaseTransformOutputParser[Dict[str, Any]]):
    """OutputParser that wraps the string output into a structured format."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this class is serializable."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "output_parser"]

    @property
    def _type(self) -> str:
        """Return the output parser type for serialization."""
        return "openai_style"

    def parse(self, text: str) -> Dict[str, Any]:
        """Returns the input text wrapped in an OpenAI-like response structure."""
        return {
            "content": text
        }