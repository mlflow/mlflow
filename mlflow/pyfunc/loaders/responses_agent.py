import warnings
from typing import Any, Generator, Optional

import pydantic

from mlflow.exceptions import MlflowException
from mlflow.models.utils import _convert_llm_ndarray_to_list
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR
from mlflow.pyfunc.model import _load_context_model_and_signature
from mlflow.types.responses import ResponsesRequest, ResponsesResponse, ResponsesStreamEvent
from mlflow.types.type_hints import model_validate
from mlflow.utils.annotations import experimental


def _load_pyfunc(model_path: str, model_config: Optional[dict[str, Any]] = None):
    _, responses_agent, _ = _load_context_model_and_signature(model_path, model_config)
    return _ResponsesAgentPyfuncWrapper(responses_agent)


@experimental
class _ResponsesAgentPyfuncWrapper:
    """
    Wrapper class that converts dict inputs to pydantic objects accepted by
    :class:`~ResponsesAgent`.
    """

    def __init__(self, responses_agent):
        self.responses_agent = responses_agent

    def get_raw_model(self):
        """
        Returns the underlying model.
        """
        return self.responses_agent

    def _convert_input(self, model_input) -> ResponsesRequest:
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
        return ResponsesRequest(**dict_input)

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

    def _response_to_dict_stream(self, response) -> dict[str, Any]:
        if isinstance(response, ResponsesStreamEvent):
            return response.model_dump_compat(exclude_none=True)
        try:
            from pydantic import TypeAdapter

            TypeAdapter(ResponsesStreamEvent).validate_python(response)
        except ImportError:
            warnings.warn(message="Pydantic>=2.0.0 is not installed. Skipping output validation.")
        except pydantic.ValidationError as e:
            raise MlflowException(
                message=(
                    f"Model returned an invalid response. "
                    f"Expected a {ResponsesStreamEvent.__name__} object or "
                    f"dictionary with the same schema. Pydantic validation error: {e}"
                ),
                error_code=INTERNAL_ERROR,
            ) from e
        return response

    def predict(self, model_input: dict[str, Any], params=None) -> dict[str, Any]:
        """
        Args:
            model_input: A dict with the
                :py:class:`ResponsesRequest <mlflow.types.responses.ResponsesRequest>` schema.
            params: Unused in this function, but required in the signature because
                `load_model_and_predict` in `utils/_capture_modules.py` expects a params field

        Returns:
            A dict with the
            (:py:class:`ResponsesResponse <mlflow.types.responses.ResponsesResponse>`)
            schema.
        """
        request = self._convert_input(model_input)
        response = self.responses_agent.predict(request)
        return self._response_to_dict(response, ResponsesResponse)

    def predict_stream(
        self, model_input: dict[str, Any], params=None
    ) -> Generator[dict[str, Any], None, None]:
        """
        Args:
            model_input: A dict with the
                :py:class:`ResponsesRequest <mlflow.types.responses.ResponsesRequest>` schema.
            params: Unused in this function, but required in the signature because
                `load_model_and_predict` in `utils/_capture_modules.py` expects a params field

        Returns:
            A generator over dicts with the
                (:py:class:`ResponsesStreamEvent <mlflow.types.responses.ResponsesStreamEvent>`)
                schema.
        """
        request = self._convert_input(model_input)
        for response in self.responses_agent.predict_stream(request):
            yield self._response_to_dict_stream(response)
