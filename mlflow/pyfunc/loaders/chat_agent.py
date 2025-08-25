from typing import Any, Generator

import pydantic

from mlflow.exceptions import MlflowException
from mlflow.models.utils import _convert_llm_ndarray_to_list
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR
from mlflow.pyfunc.model import (
    _load_context_model_and_signature,
)
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)
from mlflow.types.type_hints import model_validate


def _load_pyfunc(model_path: str, model_config: dict[str, Any] | None = None):
    _, chat_agent, _ = _load_context_model_and_signature(model_path, model_config)
    return _ChatAgentPyfuncWrapper(chat_agent)


class _ChatAgentPyfuncWrapper:
    """
    Wrapper class that converts dict inputs to pydantic objects accepted by :class:`~ChatAgent`.
    """

    def __init__(self, chat_agent):
        """
        Args:
            chat_agent: An instance of a subclass of :class:`~ChatAgent`.
        """
        self.chat_agent = chat_agent

    def get_raw_model(self):
        """
        Returns the underlying model.
        """
        return self.chat_agent

    def _convert_input(
        self, model_input
    ) -> tuple[list[ChatAgentMessage], ChatContext | None, dict[str, Any] | None]:
        import pandas

        if isinstance(model_input, dict):
            dict_input = model_input
        elif isinstance(model_input, pandas.DataFrame):
            dict_input = {
                k: _convert_llm_ndarray_to_list(v[0])
                for k, v in model_input.to_dict(orient="list").items()
            }
        else:
            raise MlflowException(
                "Unsupported model input type. Expected a dict or pandas.DataFrame, but got "
                f"{type(model_input)} instead.",
                error_code=INTERNAL_ERROR,
            )

        messages = [ChatAgentMessage(**message) for message in dict_input.get("messages", [])]
        context = ChatContext(**dict_input["context"]) if "context" in dict_input else None
        custom_inputs = dict_input.get("custom_inputs", None)

        return messages, context, custom_inputs

    def _response_to_dict(self, response, pydantic_class) -> dict[str, Any]:
        if isinstance(response, pydantic_class):
            return response.model_dump_compat(exclude_none=True)
        try:
            model_validate(pydantic_class, response)
        except pydantic.ValidationError as e:
            raise MlflowException(
                message=(
                    f"Model returned an invalid response. Expected a {pydantic_class.__name__} "
                    f"object or dictionary with the same schema. Pydantic validation error: {e}"
                ),
                error_code=INTERNAL_ERROR,
            ) from e
        return response

    def predict(self, model_input: dict[str, Any], params=None) -> dict[str, Any]:
        """
        Args:
            model_input: A dict with the
                :py:class:`ChatAgentRequest <mlflow.types.agent.ChatAgentRequest>` schema.
            params: Unused in this function, but required in the signature because
                `load_model_and_predict` in `utils/_capture_modules.py` expects a params field

        Returns:
            A dict with the (:py:class:`ChatAgentResponse <mlflow.types.agent.ChatAgentResponse>`)
                schema.
        """
        messages, context, custom_inputs = self._convert_input(model_input)
        response = self.chat_agent.predict(messages, context, custom_inputs)
        return self._response_to_dict(response, ChatAgentResponse)

    def predict_stream(
        self, model_input: dict[str, Any], params=None
    ) -> Generator[dict[str, Any], None, None]:
        """
        Args:
            model_input: A dict with the
                :py:class:`ChatAgentRequest <mlflow.types.agent.ChatAgentRequest>` schema.
            params: Unused in this function, but required in the signature because
                `load_model_and_predict` in `utils/_capture_modules.py` expects a params field

        Returns:
            A generator over dicts with the
                (:py:class:`ChatAgentChunk <mlflow.types.agent.ChatAgentChunk>`) schema.
        """
        messages, context, custom_inputs = self._convert_input(model_input)
        for response in self.chat_agent.predict_stream(messages, context, custom_inputs):
            yield self._response_to_dict(response, ChatAgentChunk)
