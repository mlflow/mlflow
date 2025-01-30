from typing import Any, Generator, Optional

import pydantic

from mlflow.exceptions import MlflowException
from mlflow.models.utils import _convert_llm_ndarray_to_list
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR
from mlflow.pyfunc.model import (
    _load_context_model_and_signature,
)
from mlflow.types.agent import ChatAgentMessage, ChatAgentParams, ChatAgentResponse
from mlflow.types.type_hints import model_validate
from mlflow.utils.annotations import experimental


def _load_pyfunc(model_path: str, model_config: Optional[dict[str, Any]] = None):
    _, chat_agent, _ = _load_context_model_and_signature(model_path, model_config)
    return _ChatAgentPyfuncWrapper(chat_agent)


@experimental
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

    # TODO: bbqiu
    def _convert_input(self, messages, params):
        import pandas

        if isinstance(messages, dict):
            dict_input = messages
        elif isinstance(messages, pandas.DataFrame):
            dict_input = {
                k: _convert_llm_ndarray_to_list(v[0])
                for k, v in messages.to_dict(orient="list").items()
            }
        else:
            raise MlflowException(
                "Unsupported model input type. Expected a dict or pandas.DataFrame, "
                f"but got {type(messages)} instead.",
                error_code=INTERNAL_ERROR,
            )

        messages = [ChatAgentMessage(**message) for message in dict_input.pop("messages", [])]
        params = ChatAgentParams(**dict_input)
        return messages, params

    def _response_to_dict(self, response: ChatAgentResponse) -> dict[str, Any]:
        try:
            model_validate(ChatAgentResponse, response)
        except pydantic.ValidationError as e:
            raise MlflowException(
                message=(
                    f"Model returned an invalid response. Expected a ChatAgentResponse object "
                    f"or dictionary with the same schema. Pydantic validation error: {e}"
                ),
                error_code=INTERNAL_ERROR,
            ) from e
        if isinstance(response, ChatAgentResponse):
            return response.model_dump_compat(exclude_none=True)
        return response

    def predict(self, model_input: dict[str, Any]) -> dict[str, Any]:
        """
        Args:
            model_input: A dict with the (:py:class:`ChatAgentRequest <mlflow.
            types.agent.ChatAgentRequest>`) schema.

        Returns:
            A dict with the (:py:class:`ChatAgentResponse <mlflow.types.agent.
            ChatAgentResponse>`) schema.
        """
        model_input = self._convert_input(model_input)
        response = self.chat_agent.predict(model_input)
        return self._response_to_dict(response)

    def predict_stream(self, model_input: dict[str, Any]) -> Generator[dict[str, Any], None, None]:
        """
        Args:
             model_input: A dict with the (:py:class:`ChatAgentRequest <mlflow.
             types.agent.ChatAgentRequest>`) schema.

         Returns:
             A generator over dicts with the (:py:class:`ChatAgentResponse <mlflow.types.agent.
             ChatAgentResponse>`) schema.
        """
        model_input = self._convert_input(model_input)
        for response in self.chat_agent.predict_stream(model_input):
            yield self._response_to_dict(response)
