from dataclasses import asdict
from typing import Any, Iterator, List

from langchain_core.output_parsers.transform import BaseTransformOutputParser
from langchain_core.messages.base import BaseMessage

from mlflow.models.rag_signatures import (
    ChainCompletionChoice,
    ChatCompletionResponse as RagChatCompletionResponse,
    Message,
    StringResponse,
)
from mlflow.types.llm import (
    ChatChoice,
    ChatChoiceDelta,
    ChatChunkChoice,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatMessage,
)
from mlflow.utils.annotations import deprecated


@deprecated("mlflow.langchain.output_parser.ChatCompletionOutputParser")
class ChatCompletionsOutputParser(BaseTransformOutputParser[dict[str, Any]]):
    """
    OutputParser that wraps the string output into a dictionary representation of a
    :py:class:`ChatCompletionResponse`
    """

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this class is serializable."""
        return True

    @property
    def _type(self) -> str:
        """Return the output parser type for serialization."""
        return "mlflow_simplified_chat_completions"

    def parse(self, text: str) -> dict[str, Any]:
        return asdict(
            RagChatCompletionResponse(
                choices=[ChainCompletionChoice(message=Message(role="assistant", content=text))],
                object="chat.completion",
            )
        )


class ChatCompletionOutputParser(BaseTransformOutputParser[str]):
    """
    OutputParser that wraps the string output into a dictionary representation of a
    :py:class:`ChatCompletionResponse` or :py:class:`ChatCompletionChunk`
    when streaming
    """

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
        return "mlflow_chat_completion"

    def parse(self, text: str) -> dict[str, Any]:
        """Returns the input text as a ChatCOmpletionResponse with no changes."""
        return ChatCompletionResponse(
            choices=[ChatChoice(message=ChatMessage(role="assistant", content=text))]
        ).to_dict()

    def transform(self, input: Iterator[BaseMessage], config, **kwargs) -> Iterator[dict[str, Any]]:
        """Returns a generator of ChatCompletionChunk objects"""
        for chunk in input:
            yield ChatCompletionChunk(
                choices=[ChatChunkChoice(delta=ChatChoiceDelta(content=chunk.content))]
            ).to_dict()


@deprecated("mlflow.langchain.output_parser.ChatCompletionOutputParser")
class StringResponseOutputParser(BaseTransformOutputParser[dict[str, Any]]):
    """
    OutputParser that wraps the string output into an dictionary representation of a
    :py:class:`StringResponse`
    """

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this class is serializable."""
        return True

    @property
    def _type(self) -> str:
        """Return the output parser type for serialization."""
        return "mlflow_simplified_str_object"

    def parse(self, text: str) -> dict[str, Any]:
        return asdict(StringResponse(content=text))
