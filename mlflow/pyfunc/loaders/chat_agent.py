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

    def _convert_input(self, model_input, params):
        import pandas

        # if the model_input and params are already Pydantic models, return
        if (
            isinstance(model_input, list)
            and model_input
            and isinstance(model_input[0], ChatAgentMessage)
            and (params is None or isinstance(params, ChatAgentParams))
        ):
            return model_input, params

        if isinstance(model_input, dict):
            dict_input = model_input
        elif isinstance(model_input, pandas.DataFrame):
            dict_input = {
                k: _convert_llm_ndarray_to_list(v[0])
                for k, v in model_input.to_dict(orient="list").items()
            }
        else:
            raise MlflowException(
                "Unsupported model input type. Expected a dict or pandas.DataFrame, "
                f"but got {type(model_input)} instead.",
                error_code=INTERNAL_ERROR,
            )

        model_input = [ChatAgentMessage(**message) for message in dict_input.pop("messages", [])]
        params = ChatAgentParams(**dict_input)
        return model_input, params

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
            return response.model_dump(exclude_none=True)
        return response

    def predict(
        self, model_input: dict[str, Any], params: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """
        Args:
            model_input: Model input data in the form of a ChatAgent request.
            params: Additional parameters to pass to the model for inference.

        Returns:
            Model predictions in :py:class:`~ChatAgentResponse` format.
        """
        model_input, params = self._convert_input(model_input, params)
        response = self.chat_agent.predict(model_input, params)
        return self._response_to_dict(response)

    def predict_stream(
        self, model_input: dict[str, Any], params: Optional[dict[str, Any]] = None
    ) -> Generator[dict[str, Any], None, None]:
        """
        Args:
            model_input: Model input data in the form of a ChatAgent request.
            params: Additional parameters to pass to the model for inference.

        Returns:
            Generator over model predictions in :py:class:`~ChatAgentResponse` format.
        """
        model_input, params = self._convert_input(model_input, params)
        for response in self.chat_agent.predict_stream(model_input, params):
            yield self._response_to_dict(response)
