import inspect
import logging
from typing import Any, Generator

from mlflow.exceptions import MlflowException
from mlflow.models.utils import _convert_llm_ndarray_to_list
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR
from mlflow.pyfunc.model import (
    _load_context_model_and_signature,
)
from mlflow.types.llm import ChatCompletionChunk, ChatCompletionResponse, ChatMessage, ChatParams

_logger = logging.getLogger(__name__)


def _load_pyfunc(model_path: str, model_config: dict[str, Any] | None = None):
    context, chat_model, signature = _load_context_model_and_signature(model_path, model_config)
    return _ChatModelPyfuncWrapper(chat_model=chat_model, context=context, signature=signature)


class _ChatModelPyfuncWrapper:
    """
    Wrapper class that converts dict inputs to pydantic objects accepted by :class:`~ChatModel`.
    """

    def __init__(self, chat_model, context, signature):
        """
        Args:
            chat_model: An instance of a subclass of :class:`~ChatModel`.
            context: A :class:`~PythonModelContext` instance containing artifacts that
                        ``chat_model`` may use when performing inference.
            signature: :class:`~ModelSignature` instance describing model input and output.
        """
        self.chat_model = chat_model
        self.context = context
        self.signature = signature

    def get_raw_model(self):
        """
        Returns the underlying model.
        """
        return self.chat_model

    def _convert_input(self, model_input):
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
                "Unsupported model input type. Expected a dict or pandas.DataFrame, "
                f"but got {type(model_input)} instead.",
                error_code=INTERNAL_ERROR,
            )

        messages = [ChatMessage.from_dict(message) for message in dict_input.pop("messages", [])]
        params = ChatParams.from_dict(dict_input)
        return messages, params

    def predict(
        self, model_input: dict[str, Any], params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Args:
            model_input: Model input data in the form of a chat request.
            params: Additional parameters to pass to the model for inference.
                       Unused in this implementation, as the params are handled
                       via ``self._convert_input()``.

        Returns:
            Model predictions in :py:class:`~ChatCompletionResponse` format.
        """
        messages, params = self._convert_input(model_input)
        parameters = inspect.signature(self.chat_model.predict).parameters
        if "context" in parameters or len(parameters) == 3:
            response = self.chat_model.predict(self.context, messages, params)
        else:
            response = self.chat_model.predict(messages, params)
        return self._response_to_dict(response)

    def _response_to_dict(self, response: ChatCompletionResponse) -> dict[str, Any]:
        if not isinstance(response, ChatCompletionResponse):
            raise MlflowException(
                "Model returned an invalid response. Expected a ChatCompletionResponse, but "
                f"got {type(response)} instead.",
                error_code=INTERNAL_ERROR,
            )
        return response.to_dict()

    def _streaming_response_to_dict(self, response: ChatCompletionChunk) -> dict[str, Any]:
        if not isinstance(response, ChatCompletionChunk):
            raise MlflowException(
                "Model returned an invalid response. Expected a ChatCompletionChunk, but "
                f"got {type(response)} instead.",
                error_code=INTERNAL_ERROR,
            )
        return response.to_dict()

    def predict_stream(
        self, model_input: dict[str, Any], params: dict[str, Any] | None = None
    ) -> Generator[dict[str, Any], None, None]:
        """
        Args:
            model_input: Model input data in the form of a chat request.
            params: Additional parameters to pass to the model for inference.
                       Unused in this implementation, as the params are handled
                       via ``self._convert_input()``.

        Returns:
            Generator over model predictions in :py:class:`~ChatCompletionChunk` format.
        """
        messages, params = self._convert_input(model_input)
        parameters = inspect.signature(self.chat_model.predict_stream).parameters
        if "context" in parameters or len(parameters) == 3:
            stream = self.chat_model.predict_stream(self.context, messages, params)
        else:
            stream = self.chat_model.predict_stream(messages, params)

        for response in stream:
            yield self._streaming_response_to_dict(response)
