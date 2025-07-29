from dataclasses import asdict
from typing import Any, Iterator
from uuid import uuid4

from langchain_core.messages.base import BaseMessage
from langchain_core.output_parsers.transform import BaseTransformOutputParser

from mlflow.models.rag_signatures import (
    ChainCompletionChoice,
    Message,
    StringResponse,
)
from mlflow.models.rag_signatures import (
    ChatCompletionResponse as RagChatCompletionResponse,
)
from mlflow.types.agent import ChatAgentChunk, ChatAgentMessage, ChatAgentResponse
from mlflow.types.llm import (
    ChatChoice,
    ChatChoiceDelta,
    ChatChunkChoice,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatMessage,
)
from mlflow.utils.annotations import deprecated, experimental


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

    @property
    def _type(self) -> str:
        """Return the output parser type for serialization."""
        return "mlflow_chat_completion"

    def parse(self, text: str) -> dict[str, Any]:
        """Returns the input text as a ChatCompletionResponse with no changes."""
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


@experimental(version="2.21.0")
class ChatAgentOutputParser(BaseTransformOutputParser[str]):
    """
    OutputParser that wraps the string output into a dictionary representation of a
    :py:class:`ChatAgentResponse <mlflow.types.agent.ChatAgentResponse>` or a
    :py:class:`ChatAgentChunk <mlflow.types.agent.ChatAgentChunk>` for easy interoperability.
    """

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this class is serializable."""
        return True

    @property
    def _type(self) -> str:
        """Return the output parser type for serialization."""
        return "mlflow_chat_agent"

    def parse(self, text: str) -> dict[str, Any]:
        """
        Returns the output text as a dictionary representation of a
        :py:class:`ChatAgentResponse <mlflow.types.agent.ChatAgentResponse>`.
        """
        return ChatAgentResponse(
            messages=[ChatAgentMessage(content=text, role="assistant", id=str(uuid4()))]
        ).model_dump_compat(exclude_none=True)

    def transform(self, input: Iterator[BaseMessage], config, **kwargs) -> Iterator[dict[str, Any]]:
        """
        Returns a generator of
        :py:class:`ChatAgentChunk <mlflow.types.agent.ChatAgentChunk>` objects
        """
        for chunk in input:
            if chunk.content:
                yield ChatAgentChunk(
                    delta=ChatAgentMessage(content=chunk.content, role="assistant", id=chunk.id)
                ).model_dump_compat(exclude_none=True)
