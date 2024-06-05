from dataclasses import asdict
from typing import Any, Dict

from langchain_core.output_parsers.transform import BaseTransformOutputParser

from mlflow.models.rag_signatures import (
    ChainCompletionChoice,
    ChatCompletionResponse,
    Message,
    StringResponse,
)
from mlflow.utils.annotations import experimental


@experimental
class ChatCompletionsOutputParser(BaseTransformOutputParser[Dict[str, Any]]):
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

    def parse(self, text: str) -> Dict[str, Any]:
        return asdict(
            ChatCompletionResponse(
                choices=[ChainCompletionChoice(message=Message(role="assistant", content=text))]
            )
        )


@experimental
class StringResponseOutputParser(BaseTransformOutputParser[Dict[str, Any]]):
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

    def parse(self, text: str) -> Dict[str, Any]:
        return asdict(StringResponse(content=text))
