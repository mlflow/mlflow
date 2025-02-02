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
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse
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


@experimental
class ChatAgentOutputParser(BaseTransformOutputParser[str]):
    """
    OutputParser that wraps the string output into a dictionary representation of a
     :py:class:`ChatAgentResponse` for easy interoperability. Use this with the helper class
     :py:class:`mlflow.langchain.chat_agent_langchain.LangChainChatAgent`.
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
        """Returns the output text as a dictionary representation of a
        :py:class:`ChatAgentResponse`.
        """
        return ChatAgentResponse(
            messages=[ChatAgentMessage(content=text, role="assistant")]
        ).model_dump_compat(exclude_none=True)

    def transform(self, input: Iterator[BaseMessage], config, **kwargs) -> Iterator[dict[str, Any]]:
        """Returns a generator of ChatAgentResponse objects"""
        uuid = str(uuid4())
        for chunk in input:
            if chunk.content:
                yield ChatAgentResponse(
                    messages=[ChatAgentMessage(content=chunk.content, role="assistant", id=uuid)]
                ).model_dump_compat(exclude_none=True)
