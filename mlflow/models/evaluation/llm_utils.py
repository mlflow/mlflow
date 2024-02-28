from abc import abstractmethod
from typing import Any, Dict, List, Optional

import requests

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.models.evaluation.base import _ModelType
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR

HINT_MSG = (
    "Invalid response format:\n{response}\n MLflow Evaluation expects OpenAI-compatible "
    "response format for chat and completion models. Please refer to OpenAI documentation: "
    "https://platform.openai.com/docs/api-reference/introduction for the expected format.\n\n"
    "HINT: If your endpoint has a different response format, you can still use  it for evaluation "
    "by defining a custom prediction function. For more details, please refer to: "
    "https://mlflow.org/docs/latest/llms/llm-evaluate/index.html#evaluating-with-a-custom-function "
)


class _ServedLLMEndpointModel(mlflow.pyfunc.PythonModel):
    """ """

    def __init__(
        self,
        endpoint: str,
        default_params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.endpoint = endpoint
        self.default_params = default_params or {}
        self.headers = headers or {}

    @abstractmethod
    def convert_input(self, input_data: str, params) -> Any:
        pass

    @abstractmethod
    def extract_output(self, response) -> str:
        pass

    def predict(self, context, model_input, params=None) -> List[str]:
        headers = {"Content-Type": "application/json", **(self.headers or {})}
        params = {**self.default_params, **(params or {})}

        if "inputs" not in model_input.columns:
            raise MlflowException(
                f"Invalid input column: {model_input.columns}. The input column for "
                "evaluating a chat or completion model endpoint must be named 'inputs'"
            )

        predictions = []
        for input_data in model_input["inputs"]:
            if not isinstance(input_data, str):
                raise MlflowException(
                    f"Invalid input column type: {type(input_data), input_data}. The input column "
                    "for evaluating a served LLM model must contain only string values."
                )

            request = self.convert_input(input_data, params)

            response = requests.post(self.endpoint, json=request, headers=headers)
            response.raise_for_status()

            text = self.extract_output(response.json())
            predictions.append(text)

        return predictions


class _ServedChatEndpointModel(_ServedLLMEndpointModel):
    def convert_input(self, input_data: str, params) -> Dict[str, Any]:
        return {
            "messages": [
                {
                    "content": input_data,
                    "role": "user",
                }
            ],
            **params,
        }

    def extract_output(self, response) -> str:
        try:
            text = response["choices"][0]["message"]["content"]
        except (KeyError, TypeError) as e:
            raise MlflowException(HINT_MSG.format(response=response)) from e
        return text


class _ServedCompletionEndpointModel(_ServedLLMEndpointModel):
    def convert_input(self, input_data: str, params) -> Dict[str, Any]:
        return {"prompt": input_data, **params}

    def extract_output(self, response) -> str:
        try:
            text = response["choices"][0]["text"]
        except TypeError as e:
            raise MlflowException(
                HINT_MSG.format(response=response), error_code=INTERNAL_ERROR
            ) from e
        return text


def get_model_from_llm_endpoint_url(
    endpoint: str,
    model_type: _ModelType,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
):
    """Returns a PythonModel instance that wraps a chat or completion model endpoint.

    Args:
        endpoint: The URL of the endpoint to use for model evaluation.
        model_type: The type of the model endpoint. Must be one of _ModelType.CHAT or
                    _ModelType.COMPLETION.
        params: Additional parameters to pass to the model for inference.
        headers: Additional headers to include in the request to the endpoint.
    """
    from mlflow.pyfunc.model import _PythonModelPyfuncWrapper

    if model_type == _ModelType.CHAT:
        python_model = _ServedChatEndpointModel(endpoint, params, headers)
    elif model_type == _ModelType.COMPLETION:
        python_model = _ServedCompletionEndpointModel(endpoint, params, headers)
    else:
        raise MlflowException(
            f"Invalid model type: '{model_type}'. Please specify {_ModelType.CHAT} or "
            f"{_ModelType.COMPLETION} to use a endpoint URL for model to evaluate."
        )

    return _PythonModelPyfuncWrapper(python_model, None, None)
